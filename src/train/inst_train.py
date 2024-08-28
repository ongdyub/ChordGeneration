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
from ..model.inst_transformer import InstTransformer, InstEncoder, InstNoPEEncoder, InstTransformer
from ..loader.inst_loader import InstDataset, create_dataloaders
import setproctitle

verbose = True
detail = False

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(device)

#############################################################
#############################################################
#############################################################
batch_size = 12
src_vocab_size = 150
tgt_vocab_size = 133
num_epochs = 500
#############################################################
#############################################################
#############################################################

folder = 'inst'
for i in range(1):
    name = f'Proj_1'
    model = InstEncoder(d_model=512, num_heads=8, d_ff=2048, num_layers=6, vocab_size=150, inst_size=133, max_len=1024)
    
    # name = f'Test'
    # model = InstEncoder(d_model=512, num_heads=8, d_ff=2048, num_layers=6, vocab_size=150, inst_size=133, max_len=1024)
    
    name = f'NoPE_1'
    model = InstNoPEEncoder(d_model=512, num_heads=8, d_ff=2048, num_layers=6, vocab_size=150, inst_size=133, max_len=1024)
    
    mode = 'Proj'
    # name = f'Proj_2'
    # model = InstEncoder(d_model=256, num_heads=16, d_ff=1024, num_layers=6, vocab_size=150, inst_size=133, max_len=1024)
    
    name = f'NoPE_2'
    model = InstNoPEEncoder(d_model=256, num_heads=16, d_ff=1024, num_layers=6, vocab_size=150, inst_size=133, max_len=1024)
    
    name = f'Trans_1'
    mode = 'Trans'
    model = InstTransformer(src_vocab_size=150, tgt_vocab_size=133)
    
    mode = 'Proj'
    name = f'Proj_3_med'
    model = InstEncoder(d_model=256, num_heads=16, d_ff=1024, num_layers=6, vocab_size=150, inst_size=133, max_len=1024)
    
    name = f'Test_trans'
    mode = 'Trans'
    model = InstTransformer(src_vocab_size=150, tgt_vocab_size=133)

#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]


# name = f'Test'
# model = InstNoPEEncoder(d_model=512, num_heads=8, d_ff=2048, num_layers=6, vocab_size=150, inst_size=133, max_len=1024)
# Optimizer
for i in range(1):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.97 ** epoch)
    pos_weight = torch.tensor([6084875,     130,     130, 6084875,       2,     114,     303,     221,
            139,     496,     107,     715,     131,      12,     199,      18,
             19,      29,      28,     429,     356,     370,     348,      67,
            276,     132,     188,     670,      48,      54,     159,      44,
            219,     126,     133,    1066,      23,      18,      32,     209,
            519,     630,     306,     388,       3,       5,       4,       4,
             71,      16,      15,       3,       1,      73,      88,     369,
             13,     229,     625,     776,       1,       2,       2,     176,
              1,      81,     244,     503,      45,       4,       5,       6,
              2,      30,       2,       1,       7,       1,     114,     160,
           1368,    1401,     231,     276,     172,     175,     337,    2100,
           4161,     722,    3894,     237,     532,     430,     873,     543,
           1465,    1522,    1063,    1501,     547,    2119,    1875,     811,
            918,    3662,    1870,    2032,     896,     206,    2539,     695,
           1234,     794,     565,    4447,     477,   15173,     741,    2570,
            452,     523,    4110,    1484,   23402,    8047,    2365,   21887,
          27911,    7672,   12918,    3877,       1]).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # med_weight = torch.full((133,), 130).to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=med_weight)

#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------


train_loader, val_loader, test_loader = create_dataloaders(batch_size)
print(f'Split Train : {len(train_loader.dataset)}, Val : {len(val_loader.dataset)}, Test : {len(test_loader.dataset)}')
print(name)

setproctitle.setproctitle(name)

if not os.path.exists(f'../../../workspace/out/{folder}/{name}'):
    os.makedirs(f'../../../workspace/out/{folder}/{name}')


def run(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, epochs=num_epochs, device=device):

    model.to(device)
    model.train()

    best_test_acc = -1
    best_val_acc = -1
    patience_counter = 0
    
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
        
        for (chords, targets, lengths) in tqdm(train_loader, ncols=60):
            optimizer.zero_grad()
            
            if mode == 'Trans':
                chords = chords.to(device)
                inputs = targets[:, :-1, :].to(device)
                targets = targets[:, 1:, :].to(device)
                
                outputs = model(chords, inputs)
                loss = criterion(outputs, targets)
            else:
                chords = chords.to(device)
                targets = targets.to(device)
                
                outputs = model(chords)
                loss = criterion(outputs, targets)
                
            mask = torch.arange(targets.size(1)).expand(len(lengths), targets.size(1)) < torch.tensor(lengths).unsqueeze(1)
            mask = mask.unsqueeze(-1).expand_as(targets).to(device)  # Shape: [batch_size, max_length, num_classes]

            # Apply mask to the loss (only consider non-padded tokens)
            loss = loss * mask
            
            # Reduce the loss across valid (non-padded) tokens
            loss = loss.sum() / mask.sum()
        
            loss.backward()
            
            
            optimizer.step()
            
            total_train_loss += loss.item()
            epoch_loss.append(loss.item())
            
            cnt, z_z, o_o, extra, lack = subset_accuracy(outputs, targets, lengths)
            
            total_cnt += cnt
            total_correct += o_o
            total_acc += (o_o/(o_o + lack + extra))
            
            if verbose:
                break
        
        scheduler.step()
        save_epoch_loss(epoch, epoch_loss)
        
        train_loss.append(total_train_loss / len(train_loader.dataset))
        train_acc.append(o_o/(o_o + lack + extra))
        print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.10f}, Accuracy: {total_correct}/{(o_o + lack + extra)} {(total_correct/(o_o + lack + extra)):.8f}, 0_0: {z_z}, 1_1: {o_o} LR: {scheduler.get_last_lr()[0]:.12f}, INFO: {name}')
        
        model.eval()
        total_val_loss = 0
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        
        with torch.no_grad():
            for (chords, targets, lengths) in tqdm(val_loader, ncols=60):
                
                if mode == 'Trans':
                    chords = chords.to(device)
                    inputs = targets[:, :-1, :].to(device)
                    targets = targets[:, 1:, :].to(device)
                    
                    outputs = model(chords, inputs)
                    loss = criterion(outputs, targets)
                else:
                    chords = chords.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(chords)
                    loss = criterion(outputs, targets)

                loss = criterion(outputs, targets)
                mask = torch.arange(targets.size(1)).expand(len(lengths), targets.size(1)) < torch.tensor(lengths).unsqueeze(1)
                mask = mask.unsqueeze(-1).expand_as(targets).to(device)  # Shape: [batch_size, max_length, num_classes]

                # Apply mask to the loss (only consider non-padded tokens)
                loss = loss * mask
                
                # Reduce the loss across valid (non-padded) tokens
                loss = loss.sum() / mask.sum()
                
                total_val_loss += loss.item()
                
                cnt, z_z, o_o, extra, lack = subset_accuracy(outputs, targets, lengths)
            
                total_cnt += cnt
                total_correct += o_o
                total_acc += (o_o/(o_o + lack + extra))
                
                if verbose:
                    break

        val_loss.append(total_val_loss / len(val_loader.dataset))
        val_acc.append(total_correct/(o_o + lack + extra))
        print(f'Epoch {epoch+1}, Val Loss: {val_loss[-1]:.10f}, 0_0: {z_z}, 1_1: {o_o}, Accuracy: {total_correct}/{(o_o + lack + extra)} {(total_correct/(o_o + lack + extra)):.8f}') 
        
        total_cnt = 0
        total_correct = 0
        total_acc = 0
        total_lack = 0
        total_extra = 0
        
        for (chords, targets, lengths) in tqdm(test_loader, ncols=60):
            if mode == 'Trans':
                chords = chords.to(device)
                inputs = targets[:, :-1, :].to(device)
                targets = targets[:, 1:, :].to(device)
                
                outputs = model(chords, inputs)
                loss = criterion(outputs, targets)
            else:
                chords = chords.to(device)
                targets = targets.to(device)
                
                outputs = model(chords)
                loss = criterion(outputs, targets)
                
            cnt, z_z, o_o, extra, lack = subset_accuracy(outputs, targets, lengths)
            
            total_cnt += cnt
            total_correct += o_o
            total_acc += (o_o/(o_o + lack + extra))

            total_cnt += cnt
            total_lack += lack
            total_extra += extra
            
            if verbose:
                break
        
        test_acc.append(total_correct/(o_o + lack + extra))
        print(f'Epoch {epoch+1}, Test Accuracy: {total_correct}/{(o_o + lack + extra)} {(total_correct/(o_o + lack + extra)):.8f}, 0_0: {z_z}, 1_1: {o_o}, Lack : {total_lack}, Extra : {total_extra}')
            
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

        if test_acc[-1] > best_test_acc or val_acc[-1] > best_val_acc:
            best_test_acc = test_acc[-1]
            best_val_acc = val_acc[-1]
            patience_counter = 0
            
            torch.save(model.state_dict(), f'../../../workspace/out/{folder}/{name}/model_{epoch+1}_{train_acc[-1]:.4f}_{val_acc[-1]:.4f}_{test_acc[-1]:.4f}.pt')

            print(f'Model saved: Epoch {epoch+1} with Train Loss: {train_loss[-1]:.6f}, Val Loss: {val_loss[-1]:.6f}')        
            print(f'Model saved: Epoch {epoch+1} with Val Loss: {val_loss[-1]:.6f}, Val Acc: {val_acc[-1]:.6f}, Test Acc: {test_acc[-1]:.6f}')
        else:
            patience_counter += 1
            
        if patience_counter == 5:
            print("Early Stop")
            print("Finish The Train Model")
      
# 일단 하나씩 돌거다
# 아웃풋 뽑고
# 유효한 길이로 먼저 자르고
# 전체 마디가 있으렌데
# 각 마디별로
# 현재 마디에 나와야 하는 악기의 수
# 정확히 맞춘 수
# 나와야하는데 못맞춘거 / 안나와야하는데 맞춘거

# def subset_accuracy(outputs, targets, lengths):
#     # outputs: [batch_size, max_length, num_classes]
#     # targets: [batch_size, max_length, num_classes]
    
#     # Apply sigmoid to outputs to get probabilities
#     probs = torch.sigmoid(outputs)  # probs: [batch_size, max_length, num_classes]
    
#     # Convert probabilities to binary predictions
#     preds = (probs > 0.5).float()  # preds: [batch_size, max_length, num_classes]

#     correct = 0
#     extra = 0
#     lack = 0
    
#     total_cnt = 0
    
#     # Per Batch
#     for idx in range(len(lengths)):
#         # Per Seq Length
#         for measure in range(lengths[idx]):
#             correct += (preds[idx, measure, :] == targets[idx, measure, :]).sum().item()
#             extra += (preds[idx, measure, :] == (targets[idx, measure, :]+1)).sum().item()
#             lack += (preds[idx, measure, :] == (targets[idx, measure, :]-1)).sum().item()
#         total_cnt += lengths[idx]
        
#         if (correct + extra + lack) != (133 * sum(lengths[:idx+1])):
#             raise Exception
#     if total_cnt != sum(lengths):
#         raise Exception
    
#     return total_cnt, correct, extra, lack

def subset_accuracy(outputs, targets, lengths):
    # Apply sigmoid to outputs to get probabilities
    probs = torch.sigmoid(outputs)  # probs: [batch_size, max_length, num_classes]
    
    # Convert probabilities to binary predictions
    preds = (probs > 0.5).float()  # preds: [batch_size, max_length, num_classes]
    
    # Create a mask based on lengths to ignore padding positions
    mask = torch.zeros_like(preds, dtype=torch.bool)
    
    for idx, length in enumerate(lengths):
        mask[idx, :length, :] = 1  # Only consider up to the length specified for each batch

    # ones_tensor = torch.ones_like(targets)
    # zeros_tensor = torch.zeros_like(targets)
    # Calculate correct, extra, and lack predictions with masking
    # correct = (preds == targets)
    # z_z = ((correct == torch.logical_not(targets)) & mask).sum().item()
    # o_o = ((correct == targets) & mask).sum().item()
    # extra = ((preds == targets + 1) & mask).sum().item()
    # lack = ((preds == targets - 1) & mask).sum().item()
    
    # Target - Pred    
    zero_zero = torch.sum(((targets == 0) & (preds == 0)) & mask).item()
    zero_one = torch.sum(((targets == 0) & (preds == 1)) & mask).item()
    one_zero = torch.sum(((targets == 1) & (preds == 0)) & mask).item()
    one_one = torch.sum(((targets == 1) & (preds == 1)) & mask).item()
    
    total_cnt = sum(lengths)
    
    return total_cnt, zero_zero, one_one, zero_one, one_zero

        
def save_epoch_loss(epoch, epoch_loss):
    if not os.path.exists(f'../../../workspace/out/{folder}/{name}/train'):
        os.makedirs(f'../../../workspace/out/{folder}/{name}/train')
    with open(f'../../../workspace/out/{folder}/{name}/train/{epoch}_train_loss.pkl', 'wb') as file:
            pickle.dump(epoch_loss, file)

run()
