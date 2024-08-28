import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from ..model.chord_model import ChordLSTM, ChordTransformer, ChordBERT, ChordBART, RQTransformer, RQ_DelayTransformer, ChordLSTM_Q
from ..loader.chord_loader import ChordDataset, create_dataloaders, BPE_Chord_Dataset
import setproctitle

verbose = False
detail = False

base_model = 'LSTM'
folder = 'chord_bpe'

# base_model = 'RQ'
# base_model = 'RQD'
# Hyper param & Dataset Setting
#############################################################
#############################################################
#############################################################
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(device)

# LSTM
batch_size = 218
batch_size = 56
# batch_size = 36
# batch_size = 2

# BERT
# batch_size = 7
# batch_size = 4


# BART
# 
# batch_size = 3

# RQ
# batch_size = 4

# RQD
# batch_size = 20

vocab_size = 10000
# vocab_size = 15000
# vocab_size = 406070

num_epochs = 100
#############################################################
#############################################################
#############################################################



train_loader, val_loader, test_loader = create_dataloaders(batch_size, base_model)
print(f'Split Train : {len(train_loader.dataset)}, Val : {len(val_loader.dataset)}, Test : {len(test_loader.dataset)}')

#############################################################
#############################################################
#############################################################
# Define Model & Utils

# model = ChordLSTM(vocab_size=vocab_size, embedding_dim=512, hidden_dim=256, num_layers=3)
# name = 'LSTM_G1_IG_FULL'
# model.load_state_dict(torch.load('/workspace/out/chord/LSTM_G1/model_45_0.005720.pt', map_location=device))

# model = ChordLSTM(vocab_size=vocab_size, embedding_dim=256, hidden_dim=128, num_layers=1)
# name = 'LSTM_G1_IG_FULL_SMALL'

# model = ChordLSTM(vocab_size=vocab_size, embedding_dim=64, hidden_dim=64, num_layers=1)
# name = 'LSTM_G1_IG_FULL_TINY'

# model = ChordLSTM(vocab_size=vocab_size, embedding_dim=2048, hidden_dim=512, num_layers=5)
# name = 'LSTM_G1_IG_FULL_LARGE'

# tmux 0
# model = ChordLSTM(vocab_size=vocab_size, embedding_dim=512, hidden_dim=256, num_layers=3)
# # model.load_state_dict(torch.load('/workspace/out/chord/LSTM_G2/model_147_0.025941.pt', map_location=device))
# name = 'LSTM_G2'

# model = ChordLSTM(vocab_size=vocab_size, embedding_dim=512, hidden_dim=256, num_layers=3)
# name = 'LSTM_G2_IG_FULL'

# model = ChordLSTM(vocab_size=vocab_size, embedding_dim=256, hidden_dim=128, num_layers=3)
# name = 'LSTM_G3_FULL'

# model = ChordLSTM(vocab_size=vocab_size, embedding_dim=256, hidden_dim=128, num_layers=3)
# name = 'LSTM_G3_STR'


# model = ChordLSTM_Q(vocab_size=vocab_size, embedding_dim=512, hidden_dim=256, num_layers=3)
# name = 'Q_STM_G1_3avg'

# tmux 2 - BERT?
# model = ChordBERT(vocab_size=vocab_size)
# name = 'BERT_G1_IG_FULL'


# model = ChordBERT(vocab_size=vocab_size)
# name = 'BERT_G2'

# model = ChordBERT(vocab_size=vocab_size)
# name = 'BERT_G2_STRIDE'



# model = ChordBERT(vocab_size=vocab_size)
# name = 'BERT_G2_IG_FULL'

# model = ChordBART(vocab_size=vocab_size)
# name = 'BART_G1_IG_FULL'

# model = ChordBART(vocab_size=vocab_size)
# name = 'BART_G2_IG_FULL'

# model = ChordBART(vocab_size=vocab_size)
# name = 'BART_G2_IG_STR'

# model = RQTransformer(
#     num_tokens = vocab_size,             # number of tokens, in the paper they had a codebook size of 16k
#     dim = 512,                      # transformer model dimension
#     max_spatial_seq_len = 768,     # maximum positions along space
#     depth_seq_len = 3,              # number of positions along depth (residual quantizations in paper)
#     spatial_layers = 6,             # number of layers for space
#     depth_layers = 2,               # number of layers for depth
#     dim_head = 64,                  # dimension per head
#     heads = 4,                      # number of attention heads
# )
# name = 'RQ_G1_D3'

# depth = 3
# model = RQ_DelayTransformer(
#     num_tokens = vocab_size,             # number of tokens, in the paper they had a codebook size of 16k
#     dim = 512,                      # transformer model dimension
#     max_spatial_seq_len = 768,     # maximum positions along space
#     depth_seq_len = depth,              # number of positions along depth (residual quantizations in paper)
#     spatial_layers = 6,             # number of layers for space
#     depth_layers = 2,               # number of layers for depth
#     dim_head = 64,                  # dimension per head
#     heads = 4,                      # number of attention heads
# )
# name = f'Del_RQ_Dep{depth}_Target_DimFix'

# depth = 3
# model = RQ_DelayTransformer(
#     num_tokens = vocab_size,             # number of tokens, in the paper they had a codebook size of 16k
#     dim = 128,                      # transformer model dimension
#     max_spatial_seq_len = 768,     # maximum positions along space
#     depth_seq_len = depth,              # number of positions along depth (residual quantizations in paper)
#     spatial_layers = 3,             # number of layers for space
#     depth_layers = 2,               # number of layers for depth
#     dim_head = 32,                  # dimension per head
#     heads = 4,                      # number of attention heads
# )
# name = f'Del_RQ_Dep{depth}_Target_DimFix_DownSize'


model = ChordLSTM(vocab_size=vocab_size, embedding_dim=512, hidden_dim=256, num_layers=3)
name = f'LSTM_BPE_V{vocab_size}'

print(name)

setproctitle.setproctitle(name)

if not os.path.exists(f'../../../workspace/out/{folder}/{name}'):
    os.makedirs(f'../../../workspace/out/{folder}/{name}')


optimizer = model.optimizer
scheduler = model.scheduler
criterion = nn.CrossEntropyLoss(ignore_index = 0)

# Check if CUDA is available and wrap the model with DataParallel
# if torch.cuda.is_available():
#     os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
#     model.cuda()  # Move the model to GPU before wrapping with DataParallel
#     device_ids = [0, 1, 2, 3]
#     model = nn.DataParallel(model, device_ids=device_ids)
# else:
#     print("CUDA is not available. Training on CPU.")

#############################################################
#############################################################
#############################################################

def run_LSTM(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, epochs=num_epochs, device=device):

    model.to(device)
    model.train()

    best_test_acc = -1
    patience_counter = 0
    train_loss = []
    val_loss = []
    
    train_acc = []
    val_acc = []
    test_acc = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        
        epoch_loss = []
        
        for (inputs, targets) in tqdm(train_loader, ncols=60):
            if inputs.size(0) == 0:
                print("Skip Zero-Size")
                continue
            optimizer.zero_grad()
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            output_ids = torch.argmax(outputs, dim=2)
            
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Stopping training, encountered {loss.item()} loss at epoch {epoch}")
                loss = torch.clamp(loss, min=0.0001)
                continue
            
            loss.backward()
            optimizer.step()
            
            total, correct, acc = cal_acc(output_ids, targets)
            total_cnt += total
            total_correct += correct
            total_acc += acc

            total_train_loss += loss.item()
            epoch_loss.append(loss.item())
            
            if verbose:
                if detail:
                    print("Train")
                    print(inputs[:,:5])
                    print(targets[:,:5])
                    print(outputs.shape)
                    print(targets.shape)
                    print(targets.view(-1).shape)
                break
        
        save_epoch_loss(epoch, epoch_loss)    
        scheduler.step()
        
        train_loss.append(total_train_loss / len(train_loader.dataset))
        train_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}, LR: {scheduler.get_last_lr()[0]:.6f}, INFO: {name}')
        
        # continue
    
        model.eval()
        total_val_loss = 0
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        
        with torch.no_grad():
            for (inputs, targets) in tqdm(val_loader, ncols=60):
                if inputs.size(0) == 0:
                    print("Skip Zero-Size")
                    continue
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                output_ids = torch.argmax(outputs, dim=2)
                
                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Stopping training, encountered NaN/Inf loss at epoch {epoch}")
                    continue
                
                total, correct, acc = cal_acc(output_ids, targets)
                total_cnt += total
                total_correct += correct
                total_acc += acc
                
                total_val_loss += loss.item()
                
                if verbose:
                    if detail:
                        print(outputs.shape)
                        print(targets.shape)
                        print(targets.view(-1).shape)
                    break

        val_loss.append(total_val_loss / len(val_loader.dataset))
        val_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Val Loss: {val_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}') 
        
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        
        for (inputs, targets) in tqdm(test_loader, ncols=60):
            if inputs.size(0) == 0:
                print("Skip Zero-Size")
                continue
            seq_len = inputs.shape[1]

            if seq_len < 8:
                input_len = 4
                max_len = seq_len
            elif seq_len < 100:
                input_len = 8
                max_len = seq_len
            else:
                input_len = 8
                max_len = 100
                
            inputs = inputs[:,:input_len].to(device)
            targets = targets[:,:max_len].to(device)

            out = model.infer(inputs, length=max_len)
            
            out = out[:,1:max_len+1]
            
            total, correct, acc = cal_acc(out, targets)

            total_cnt += total
            total_correct += correct
            total_acc += acc
            
            if verbose:
                break
        
        test_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Test Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}')
            
        with open(f'../../../workspace/out/{folder}/{name}/train_loss.pkl', 'wb') as file:
            pickle.dump(train_loss, file)
        with open(f'../../../workspace/out/{folder}/{name}/val_loss.pkl', 'wb') as file:
            pickle.dump(val_loss, file)
        with open(f'../../../workspace/out/{folder}/{name}/train_acc.pkl', 'wb') as file:
            pickle.dump(train_acc, file)
        with open(f'../../../workspace/out/{folder}/{name}/val_acc.pkl', 'wb') as file:
            pickle.dump(val_acc, file)
        with open(f'../../../workspace/out/{folder}/{name}/test_acc.pkl', 'wb') as file:
            pickle.dump(test_acc, file)

        if test_acc[-1] > best_test_acc or train_acc[-1] > 0.8:
            best_test_acc = test_acc[-1]
            patience_counter = 0

            torch.save(model.state_dict(), f'../../../workspace/out/{folder}/{name}/model_{epoch+1}_{val_acc[-1]:.4f}_{test_acc[-1]:.4f}.pt')
            print(f'Model saved: Epoch {epoch+1} with Val Loss: {val_loss[-1]:.6f}, Val Acc: {val_acc[-1]:.6f}, Test Acc: {test_acc[-1]:.6f}')        
        else:
            patience_counter += 1
            
        if patience_counter == 5:
            print("Early Stop")
            print("Finish The Train Model")
            # break
      

def cal_acc(infer, target):
    if infer.shape[0] != target.shape[0] or infer.shape[1] != target.shape[1]:
        raise Exception
    
    # if verbose:
    #     if detail:
    #         print("CAL ACC")
    #         print(infer.shape)
    #         print(target.shape)
    
    correct = 0
    total = 0
    
    for i in range(infer.shape[0]):
            
        # zero_infer = torch.where(infer[i] == 0)[0]
        # zero_target = torch.where(target[i] == 0)[0]
        zero_infer = torch.where(infer[i] == 1)[0]
        zero_target = torch.where(target[i] == 1)[0]
        # print(zero_infer)
        # print(zero_target)
        # print(infer[i].shape[0])
        # print("[[[[[[[[[]]]]]]]]]")
        # zero_infer = zero_infer[0].item() if len(zero_infer) > 0 else infer[i].shape[0]
        zero_target = zero_target[0].item() if len(zero_target) > 0 else target[i].shape[0]
        
        # max_zero = max(zero_infer, zero_target)
        max_zero = zero_target
        
        # if verbose:
        #     if detail:
        #         # print("IN FOR")
        #         # print(infer[i].shape)
        #         # print(infer[i][:10])
        #         # print(infer[i][-10:])
        #         # print(target[i].shape)
        #         # print(target[i][:10])
        #         # print(target[i][-10:])
        #         print(zero_infer)
        #         print(zero_target)
        #         print(max_zero)
        
        infer_slice = infer[i][:max_zero]
        target_slice = target[i][:max_zero]
        
        correct += (infer_slice == target_slice).sum().item()
        total += max_zero
    if total == 0.0 or total == 0:
        total = 1

    return total, correct, correct/total
        
        
def run_RQ(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, epochs=num_epochs, device=device):

    model.to(device)
    model.train()

    best_test_acc = -1
    patience_counter = 0
    train_loss = []
    val_loss = []
    
    train_acc = []
    val_acc = []
    test_acc = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        
        epoch_loss = []
        
        for (inputs) in tqdm(train_loader, ncols=60):
            if inputs.size(0) == 0:
                print("Skip Zero-Size")
                continue
            optimizer.zero_grad()
            
            inputs = inputs.to(device)
            
            _, logits = model(inputs, return_loss = True)
            
            output_ids = torch.argmax(logits, dim=1)
            output_ids = rearrange(output_ids, 'b (s d) -> b s d', d=depth)[:,:,0]
            real_inputs = inputs[:,:,0]
            logits = logits[:,:,::depth]
            # print(logits.shape)
            # print(real_inputs.shape)
            loss = criterion(logits, real_inputs)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Stopping training, encountered {loss.item()} loss at epoch {epoch}")
                loss = torch.clamp(loss, min=0.0001)
                continue
            loss.backward()
            optimizer.step()
            
            # total, correct, acc = cal_acc(output_ids, inputs)
            total, correct, acc = cal_acc(output_ids, real_inputs)
            total_cnt += total
            total_correct += correct
            total_acc += acc

            total_train_loss += loss.item()
            epoch_loss.append(loss.item())
            
            if verbose:
                if detail:
                    print("Train")
                    print(inputs[:,:5])
                    # print(targets[:,:5])
                    # print(outputs.shape)
                    # print(targets.shape)
                    # print(targets.view(-1).shape)
                break
        
        save_epoch_loss(epoch, epoch_loss)    
        scheduler.step()
        
        train_loss.append(total_train_loss / len(train_loader.dataset))
        train_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}, LR: {scheduler.get_last_lr()[0]:.6f}, INFO: {name}')
        
        # continue
    
        model.eval()
        total_val_loss = 0
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        
        with torch.no_grad():
            for (inputs) in tqdm(val_loader, ncols=60):
                if inputs.size(0) == 0:
                    print("Skip Zero-Size")
                    continue
                inputs = inputs.to(device)
                
                _, logits = model(inputs, return_loss = True)
            
                output_ids = torch.argmax(logits, dim=1)
                output_ids = rearrange(output_ids, 'b (s d) -> b s d', d=depth)[:,:,0]
                real_inputs = inputs[:,:,0]
                logits = logits[:,:,::depth]
                # print(logits.shape)
                # print(real_inputs.shape)
                loss = criterion(logits, real_inputs)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Stopping training, encountered {loss.item()} loss at epoch {epoch}")
                    loss = torch.clamp(loss, min=0.0001)
                    continue
                
                # total, correct, acc = cal_acc(output_ids, inputs)
                total, correct, acc = cal_acc(output_ids, real_inputs)
                total_cnt += total
                total_correct += correct
                total_acc += acc
                
                total_val_loss += loss.item()
                
                if verbose:
                    if detail:
                        print(outputs.shape)
                        print(targets.shape)
                        print(targets.view(-1).shape)
                    break

        val_loss.append(total_val_loss / len(val_loader.dataset))
        val_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Val Loss: {val_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}') 
        
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        
        for (inputs) in tqdm(test_loader, ncols=60):
            inputs = rearrange(inputs, 'b s d -> b (s d)', d=depth)
            
            if inputs.size(0) == 0:
                print("Skip Zero-Size")
                continue
            seq_len = inputs.shape[1]//depth

            if seq_len < 8:
                input_len = 4
                max_len = seq_len
            elif seq_len < 100:
                input_len = 8
                max_len = seq_len
            else:
                input_len = 8
                max_len = 100
 
            infer = inputs[:,:input_len*depth].to(device)
            out = model.generate(prime=infer, length=max_len*depth)
            out = out[:,:,0]
            out = out[:,:max_len]
            inputs = inputs[:,::depth].to(device)
            inputs = inputs[:,:max_len]
            # inputs = inputs[:,:,0].to(device)
            # inputs = inputs[:,:max_len*3]
            
            total, correct, acc = cal_acc(out, inputs)

            total_cnt += total
            total_correct += correct
            total_acc += acc
            
            if verbose:
                break
        
        test_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Test Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}')
            
        with open(f'../../../workspace/out/chord/{name}/train_loss.pkl', 'wb') as file:
            pickle.dump(train_loss, file)
        with open(f'../../../workspace/out/chord/{name}/val_loss.pkl', 'wb') as file:
            pickle.dump(val_loss, file)
        with open(f'../../../workspace/out/chord/{name}/train_acc.pkl', 'wb') as file:
            pickle.dump(train_acc, file)
        with open(f'../../../workspace/out/chord/{name}/val_acc.pkl', 'wb') as file:
            pickle.dump(val_acc, file)
        with open(f'../../../workspace/out/chord/{name}/test_acc.pkl', 'wb') as file:
            pickle.dump(test_acc, file)

        if test_acc[-1] > best_test_acc or train_acc[-1] > 0.8:
            best_test_acc = test_acc[-1]
            patience_counter = 0

            torch.save(model.state_dict(), f'../../../workspace/out/chord/{name}/model_{epoch+1}_{val_acc[-1]:.4f}_{test_acc[-1]:.4f}.pt')
            print(f'Model saved: Epoch {epoch+1} with Val Loss: {val_loss[-1]:.6f}, Val Acc: {val_acc[-1]:.6f}, Test Acc: {test_acc[-1]:.6f}')        
        else:
            patience_counter += 1
            
        if patience_counter == 5:
            print("Early Stop")
            print("Finish The Train Model")
            # break
      

def cal_acc(infer, target):
    if infer.shape[0] != target.shape[0] or infer.shape[1] != target.shape[1]:
        raise Exception
    
    # if verbose:
    #     if detail:
    #         print("CAL ACC")
    #         print(infer.shape)
    #         print(target.shape)
    
    correct = 0
    total = 0
    
    for i in range(infer.shape[0]):
            
        # zero_infer = torch.where(infer[i] == 0)[0]
        # zero_target = torch.where(target[i] == 0)[0]
        zero_infer = torch.where(infer[i] == 1)[0]
        zero_target = torch.where(target[i] == 1)[0]
        # print(zero_infer)
        # print(zero_target)
        # print(infer[i].shape[0])
        # print("[[[[[[[[[]]]]]]]]]")
        # zero_infer = zero_infer[0].item() if len(zero_infer) > 0 else infer[i].shape[0]
        zero_target = zero_target[0].item() if len(zero_target) > 0 else target[i].shape[0]
        
        # max_zero = max(zero_infer, zero_target)
        max_zero = zero_target
        
        # if verbose:
        #     if detail:
        #         # print("IN FOR")
        #         # print(infer[i].shape)
        #         # print(infer[i][:10])
        #         # print(infer[i][-10:])
        #         # print(target[i].shape)
        #         # print(target[i][:10])
        #         # print(target[i][-10:])
        #         print(zero_infer)
        #         print(zero_target)
        #         print(max_zero)
        
        infer_slice = infer[i][:max_zero]
        target_slice = target[i][:max_zero]
        
        correct += (infer_slice == target_slice).sum().item()
        total += max_zero
    if total == 0.0 or total == 0:
        total = 1

    return total, correct, correct/total 
        
def save_epoch_loss(epoch, epoch_loss):
    if not os.path.exists(f'../../../workspace/out/{folder}/{name}/train'):
        os.makedirs(f'../../../workspace/out/{folder}/{name}/train')
    with open(f'../../../workspace/out/{folder}/{name}/train/{epoch}_train_loss.pkl', 'wb') as file:
            pickle.dump(epoch_loss, file)


if base_model == 'LSTM':      
    run_LSTM()
else:
    run_RQ()
    
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask