import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import T5Config, T5ForConditionalGeneration, AdamW
from transformers import BertConfig, BertModel, BertLMHeadModel, BartConfig, BartModel

class Chord_Note_LSTM(nn.Module):
    def __init__(self, chord_size=150, note_size=832, chord_dim=512, note_dim=512, num_note=2, hidden_dim=512, num_layers=5):
        super(Chord_Note_LSTM, self).__init__()
        self.chord_embedding = nn.Embedding(chord_size, chord_dim, padding_idx=0)
        self.note_embedding = nn.Embedding(note_size, note_dim, padding_idx=0)
        self.num_note = num_note
        self.lstm = nn.LSTM(chord_dim+(note_dim*num_note), hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, chord_size)
        
        self.epsilon = 1e-9
        self.optimizer = optim.AdamW(self.parameters(), lr=0.01, eps=1e-9)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=0.001)

    def forward(self, note, chord):
        chord_embed = self.chord_embedding(chord)
        note_embed = self.note_embedding(note)
        
        # note_mask = note != 0
        # note_embed = note_embed * note_mask.unsqueeze(-1).float()
        # valid_counts = note_mask.sum(dim=1, keepdim=True)
        # note_embed = note_embed.sum(dim=1) / valid_counts.clamp(min=1)
        
        bsz, seq_length, depth, embed_dim = note_embed.shape
        # assert depth == self.num_note
        note_embed = note_embed.mean(dim=2)
        # note_embed = note_embed.view(bsz, seq_length, depth * embed_dim)
        
        input_embed = torch.cat([chord_embed, note_embed], dim=2)
        
        output, (hidden, cell) = self.lstm(input_embed)
        
        logits = self.fc(output)
        
        return logits
    
    def infer(self, note, chord, length=2048):
        if len(chord.shape) == 1:
            chord = chord.unsqueeze(0)
        if len(chord.shape) > 2:
            raise Exception
        
        
        if length > 2048:
            print("Max Length is 2048. Change Length Auto to 2048")
            length = 2048
        
        with torch.no_grad():
            for step in range(length):
                chord_length = chord.shape[1]
                output = self.forward(note[:,:chord_length], chord)
                output = torch.argmax(output, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((chord, predict), dim=-1)

                chord = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids
    
    
class Chord_Note_GRU(nn.Module):
    def __init__(self, chord_size=150, note_size=832, chord_dim=512, note_dim=512, hidden_dim=512, num_layers=5):
        super(Chord_Note_GRU, self).__init__()
        self.chord_embedding = nn.Embedding(chord_size, chord_dim, padding_idx=0)
        self.note_embedding = nn.Embedding(note_size, note_dim, padding_idx=0)

        self.lstm = nn.GRU(chord_dim+note_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, chord_size)
        
        self.epsilon = 1e-9
        self.optimizer = optim.AdamW(self.parameters(), lr=0.01, eps=1e-9)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.97 ** epoch)
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=0.001)

    def forward(self, note, chord):
        chord_embed = self.chord_embedding(chord)
        note_embed = self.get_accumulated_note_embedding(note)
        
        # bsz, seq_length, depth, embed_dim = note_embed.shape
        # assert depth == self.num_note
        # note_embed = note_embed.mean(dim=2)
        # note_embed = note_embed.view(bsz, seq_length, depth * embed_dim)
        
        input_embed = torch.cat([chord_embed, note_embed], dim=2)
        
        output, hidden = self.lstm(input_embed)
        
        logits = self.fc(output)
        
        return logits
    
    def get_accumulated_note_embedding(self, note):
        batch_note_embed = []
        for i in note:
            note_embed = []
            for j in i:
                
                zero_indices = torch.where(j == 0)[0]
                if len(zero_indices) == 0:
                    zero_index = len(j)
                    # print(len(j))
                else:
                    zero_index = zero_indices[0]
                    if zero_index == 0:
                        zero_index = 1
                v_n = j[:zero_index]
                # print(v_n)
                v_n = self.note_embedding(v_n)
                # print(v_n.shape)
                v_n = v_n.mean(dim=0, keepdim=True)
                # print(v_n[:,:20])
                # print(v_n.shape)
                note_embed.append(v_n)
            note_embed = torch.cat(note_embed, dim=0)
            # print(note_embed.shape)
            batch_note_embed.append(note_embed)
        batch_note_embed = torch.stack(batch_note_embed, dim=0)
        # print(batch_note_embed.shape)
        return batch_note_embed
    
    def infer(self, note, chord, length=2048):
        if len(chord.shape) == 1:
            chord = chord.unsqueeze(0)
        if len(chord.shape) > 2:
            raise Exception
        
        
        if length > 2048:
            print("Max Length is 2048. Change Length Auto to 2048")
            length = 2048
        
        with torch.no_grad():
            for step in range(length):
                chord_length = chord.shape[1]
                output = self.forward(note[:,:chord_length], chord)
                output = torch.argmax(output, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((chord, predict), dim=-1)

                chord = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids
    
    

class Chord_Note_Conv(nn.Module):
    def __init__(self, chord_size=150, note_size=832, chord_dim=256, note_dim=256):
        super(Chord_Note_Conv, self).__init__()
        self.chord_embedding = nn.Embedding(chord_size, chord_dim, padding_idx=0)
        self.note_embedding = nn.Embedding(note_size, note_dim, padding_idx=0)
        
        self.conv1 = nn.Conv1d(chord_dim+note_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv1d(256, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, chord_size)
        
        self.epsilon = 1e-9
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-9)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=0.001)

    def forward(self, note, chord):
        chord_embed = self.chord_embedding(chord)
        note_embed = self.get_accumulated_note_embedding(note)
        
        bsz, seq_length, embed_dim = note_embed.shape
        assert seq_length == chord_embed.shape[1]
        
        input_embed = torch.cat([chord_embed, note_embed], dim=2)
        input_embed = input_embed.transpose(1, 2)
        x = F.relu(self.conv1(input_embed))

        x = F.relu(self.conv2(x))
        
        # x = F.relu(self.conv3(x))

        # x = self.conv4(x)
        x = x.transpose(1,2)
        logits = self.fc(x)
        return logits
    
    def get_accumulated_note_embedding(self, note):
        batch_note_embed = []
        for i in note:
            note_embed = []
            for j in i:
                
                zero_indices = torch.where(j == 0)[0]
                if len(zero_indices) == 0:
                    zero_index = len(j)
                    # print(len(j))
                else:
                    zero_index = zero_indices[0]
                    if zero_index == 0:
                        zero_index = 1
                v_n = j[:zero_index]
                # print(v_n)
                v_n = self.note_embedding(v_n)
                # print(v_n.shape)
                v_n = v_n.mean(dim=0, keepdim=True)
                # print(v_n[:,:20])
                # print(v_n.shape)
                note_embed.append(v_n)
            note_embed = torch.cat(note_embed, dim=0)
            # print(note_embed.shape)
            batch_note_embed.append(note_embed)
        batch_note_embed = torch.stack(batch_note_embed, dim=0)
        # print(batch_note_embed.shape)
        return batch_note_embed
        
    
    def infer(self, note, chord, length=2048):
        if len(chord.shape) == 1:
            chord = chord.unsqueeze(0)
        if len(chord.shape) > 2:
            raise Exception
        
        
        if length > 2048:
            print("Max Length is 2048. Change Length Auto to 2048")
            length = 2048
        
        with torch.no_grad():
            for step in range(length):
                chord_length = chord.shape[1]
                output = self.forward(note[:,:chord_length], chord)
                output = torch.argmax(output, dim=2)

                predict = output[:,-1].unsqueeze(1)
                output_ids = torch.cat((chord, predict), dim=-1)

                chord = output_ids
                
                # if torch.all(predict.eq(0)):
                #     break
                
                if output_ids.shape[1] > 2048:
                    break

        return output_ids
