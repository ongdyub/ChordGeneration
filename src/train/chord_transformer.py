import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.distributed as dist
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from ..model.transformer import TransformerDecoder, GPT2Model, TransformerDecoder_NOPE
from ..loader.chord_loader import ChordDataset, create_BPE_dataloaders, BPE_Chord_Dataset
import setproctitle

verbose = False
detail = False

device = torch.device("cuda:14" if torch.cuda.is_available() else "cpu")
print(device)

#############################################################
#############################################################
#############################################################
batch_size = 3
vocab_size = 150
num_epochs = 500
#############################################################
#############################################################
#############################################################


#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
base_model = 'GPT2'
folder = 'chord_bpe'
#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
# model = GPT2Model(vocab_size=vocab_size)
name = f'{base_model}_BPE_V{vocab_size}'
#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

for i in range(1):
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    base_model = 'Decoder_Fast'
    folder = 'chord_bpe'
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    # Parameters
    vocab_size = 150  # size of the vocabulary
    d_model = 512  # dimension of model
    num_layers = 6  # number of decoder layers
    num_heads = 8  # number of attention heads
    d_ff = 2048  # dimension of feed-forward network
    dropout = 0.1  # dropout rate
    model = TransformerDecoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, device=device)
    name = f'{base_model}_BPE_V{vocab_size}'
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    
    base_model = 'Decoder_Fast_NOPE'
    folder = 'chord_bpe'
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    # Parameters
    vocab_size = 150  # size of the vocabulary
    d_model = 512  # dimension of model
    num_layers = 6  # number of decoder layers
    num_heads = 8  # number of attention heads
    d_ff = 2048  # dimension of feed-forward network
    dropout = 0.1  # dropout rate
    model = TransformerDecoder_NOPE(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, device=device)
    name = f'{base_model}_BPE_V{vocab_size}'

    # Parameters
    # vocab_size = 150  # size of the vocabulary
    # d_model = 768  # dimension of model
    # num_layers = 12  # number of decoder layers
    # num_heads = 12  # number of attention heads
    # d_ff = 2048  # dimension of feed-forward network
    # dropout = 0.1  # dropout rate
    # model = TransformerDecoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, device=device)
    # name = f'{base_model}_Large_GPT_BPE_V{vocab_size}'
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]


    # Parameters
    # base_model = 'GPT2_Layer'
    # folder = 'chord_bpe'

    # vocab_size = 1000
    # n_embd=768
    # n_layer=18
    # n_head=12

    # model = GPT2Model(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head)
    # name = f'{base_model}_BPE_V{vocab_size}'
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

    # Parameters
    # base_model = 'GPT2_Dim'
    # folder = 'chord_bpe'

    # vocab_size = 1000
    # n_embd=1152
    # n_layer=12
    # n_head=12

    # model = GPT2Model(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head)
    # name = f'{base_model}_BPE_V{vocab_size}'
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]


    # Parameters
    # base_model = 'GPT2_XL'
    # folder = 'chord_bpe'

    # vocab_size = 1000
    # n_embd=512
    # n_layer=24
    # n_head=16

    # model = GPT2Model(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head)
    # name = f'{base_model}_BPE_V{vocab_size}'
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]


# Parameters
# base_model = 'GPT2_M'
# folder = 'chord_bpe'

# vocab_size = 1000
# n_embd=1024
# # 원래는 24지만 OOM으로 조금 줄임
# n_layer=16
# n_head=16

# model = GPT2Model(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head)
# name = f'{base_model}_BPE_V{vocab_size}'
#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]


# # Parameters
# base_model = 'GPT2_M'
# folder = 'chord_bpe'

# vocab_size = 10000
# n_embd=1024
# # 원래는 24지만 OOM으로 조금 줄임
# n_layer=16
# n_head=16

# model = GPT2Model(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head)
# name = f'{base_model}_BPE_V{vocab_size}'
#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

# Parameters
# base_model = 'GPT2_M'
# folder = 'chord_bpe'

# vocab_size = 20000
# n_embd=1024
# # 원래는 24지만 OOM으로 조금 줄임
# n_layer=16
# n_head=16

# model = GPT2Model(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head)
# name = f'{base_model}_BPE_V{vocab_size}'
# #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]



# Optimizer
optimizer = optim.Adam(model.parameters(), lr=3e-5)
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.97 ** epoch)

criterion = nn.CrossEntropyLoss(ignore_index = 0)

#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------


train_loader, val_loader, test_loader = create_BPE_dataloaders(batch_size, base_model)
print(f'Split Train : {len(train_loader.dataset)}, Val : {len(val_loader.dataset)}, Test : {len(test_loader.dataset)}')
print(name)

setproctitle.setproctitle(name)

if not os.path.exists(f'../../../workspace/out/{folder}/{name}'):
    os.makedirs(f'../../../workspace/out/{folder}/{name}')


def run(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, epochs=num_epochs, device=device):

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
            optimizer.zero_grad()

            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # GPT2
            # outputs = model(input_ids=inputs, labels=inputs)
            # output_ids = torch.argmax(outputs.logits, dim=2)
            # loss = outputs.loss
            
            # Decoder
            outputs = model(inputs[:,:-1])
            output_ids = torch.argmax(outputs, dim=2)
            loss = criterion(outputs.view(-1, vocab_size), targets[:,1:].reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            # GPT2
            # total, correct, acc = cal_acc(output_ids[:,:-1], targets[:,1:])
            
            # Decoder
            total, correct, acc = cal_acc(output_ids[:,:], targets[:,1:])
            
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
        
        scheduler.step()
        save_epoch_loss(epoch, epoch_loss)
        
        train_loss.append(total_train_loss / len(train_loader.dataset))
        train_acc.append(total_correct/total_cnt)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.10f}, Accuracy: {total_correct}/{total_cnt} {(total_correct/total_cnt):.8f}, LR: {scheduler.get_last_lr()[0]:.12f}, INFO: {name}')
        
        # continue
    
        model.eval()
        total_val_loss = 0
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        
        with torch.no_grad():
            for (inputs, targets) in tqdm(val_loader, ncols=60):

                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # GPT2
                # outputs = model(input_ids=inputs, labels=inputs)
                # output_ids = torch.argmax(outputs.logits, dim=2)
                # loss = outputs.loss
                
                # Decoder
                outputs = model(inputs[:,:-1])
                output_ids = torch.argmax(outputs, dim=2)
                loss = criterion(outputs.view(-1, vocab_size), targets[:,1:].reshape(-1))
                
                # GPT2
                # total, correct, acc = cal_acc(output_ids[:,:-1], targets[:,1:])
            
                # Decoder
                total, correct, acc = cal_acc(output_ids[:,:], targets[:,1:])
                
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
            seq_len = inputs.shape[1]

            if seq_len < 12:
                input_len = 4
                max_len = seq_len
            elif seq_len < 100:
                input_len = 12
                max_len = seq_len
            else:
                input_len = 12
                max_len = 100
                
            inputs = inputs[:,:input_len].to(device)
            targets = targets[:,:max_len].to(device)
            
            out = model.infer(inputs, length=max_len)
            out = out[:,:max_len]

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
      

def cal_acc(infer, target):
    if infer.shape[0] != target.shape[0] or infer.shape[1] != target.shape[1]:
        raise Exception
    correct = 0
    total = 0
    
    for i in range(infer.shape[0]):
        zero_infer = torch.where(infer[i] == 1)[0]
        zero_target = torch.where(target[i] == 1)[0]
        zero_target = zero_target[0].item() if len(zero_target) > 0 else target[i].shape[0]

        max_zero = zero_target
        
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

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask



run()
