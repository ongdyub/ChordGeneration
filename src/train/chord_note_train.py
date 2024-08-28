import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from ..model.chord_note_model import Chord_Note_LSTM, Chord_Note_GRU, Chord_Note_Conv
from ..loader.all_loader import create_dataloaders

# Hyper param & Dataset Setting
#############################################################
#############################################################
#############################################################

verbose = False
detail = False

base_model = 'GRU'
folder = 'chord_note'


device = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
print(device)

# LSTM
batch_size = 128
vocab_size = 150
num_epochs = 100
n_notes = 8


train_loader, val_loader, test_loader = create_dataloaders(batch_size, base_model, n_notes)
print(f'Split Train : {len(train_loader.dataset)}, Val : {len(val_loader.dataset)}, Test : {len(test_loader.dataset)}')


# model = ChordLSTM(vocab_size=vocab_size, embedding_dim=512, hidden_dim=256, num_layers=3)
# name = 'LSTM_G1_IG_FULL_REPRO'
# model.load_state_dict(torch.load('/workspace/out/chord/LSTM_G1/model_45_0.005720.pt', map_location=device))

# model = Chord_Note_LSTM(chord_dim=256, note_dim=128, num_note=2, hidden_dim=256, num_layers=3)
# name = 'LSTM_2Note'

# model = Chord_Note_LSTM(chord_dim=256, note_dim=64, num_note=4, hidden_dim=256, num_layers=3)
# name = 'LSTM_4Note'
# print(name)

# model = Chord_Note_LSTM(chord_dim=256, note_dim=256, num_note=1, hidden_dim=256, num_layers=3)
# name = 'LSTM_4Note_Avg'
# print(name)

# model = Chord_Note_GRU(chord_dim=32, note_dim=32, num_note=n_notes, hidden_dim=128, num_layers=3)
# name = f'GRU_{n_notes}Note_dup'
# print(name)

model = Chord_Note_GRU(chord_dim=64, note_dim=64, hidden_dim=64, num_layers=3)
name = f'GRU_{n_notes}Note_Uniq'
# 8
print(name)


# model = Chord_Note_Conv()
# name = f'Conv_{n_notes}Note_Uniq'
# print(name)

if not os.path.exists(f'../../../workspace/out/{folder}/{name}'):
    os.makedirs(f'../../../workspace/out/{folder}/{name}')

optimizer = model.optimizer
scheduler = model.scheduler
criterion = nn.CrossEntropyLoss(ignore_index = 0)

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
        
        for (chord_input, chord_targets, note) in tqdm(train_loader, ncols=60):
            optimizer.zero_grad()
            
            chord_input = chord_input.to(device)
            chord_targets = chord_targets.to(device)
            note = note.to(device)
            
            outputs = model(note, chord_input)
            output_ids = torch.argmax(outputs, dim=2)
            
            loss = criterion(outputs.view(-1, vocab_size), chord_targets.view(-1))
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Stopping training, encountered {loss.item()} loss at epoch {epoch}")
                loss = torch.clamp(loss, min=0.0001)
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 15)
            optimizer.step()
            
            total, correct, acc = cal_acc(output_ids, chord_targets)
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
            for (chord_inputs, chord_targets, note) in tqdm(val_loader, ncols=60):
                chord_inputs = chord_inputs.to(device)
                chord_targets = chord_targets.to(device)
                note = note.to(device)

                outputs = model(note, chord_inputs)
                output_ids = torch.argmax(outputs, dim=2)
                
                loss = criterion(outputs.view(-1, vocab_size), chord_targets.view(-1))
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Stopping training, encountered NaN/Inf loss at epoch {epoch}")
                    continue
                
                total, correct, acc = cal_acc(output_ids, chord_targets)
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
        
        for (chord_inputs, chord_targets, note) in tqdm(test_loader, ncols=60):
            seq_len = chord_inputs.shape[1]

            if seq_len < 8:
                input_len = 4
                max_len = seq_len
            elif seq_len < 100:
                input_len = 8
                max_len = seq_len
            else:
                input_len = 8
                max_len = 100
                
            chord_inputs = chord_inputs[:,:input_len].to(device)
            chord_targets = chord_targets[:,:max_len].to(device)
            note = note.to(device)

            out = model.infer(note, chord_inputs, length=max_len)
            
            out = out[:,1:max_len+1]
            
            total, correct, acc = cal_acc(out, chord_targets)

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
        



run_LSTM()