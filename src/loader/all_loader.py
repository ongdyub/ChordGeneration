import torch
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class SymDataset(Dataset):
    def __init__(self, data, base_model, n_notes):
        super().__init__()
        self.data = data
        self.base_model = base_model
        self.n_notes = n_notes
        
        chord_vocab_path = '/workspace/data/vocabs/chord.json'
        with open(chord_vocab_path, 'r') as file:
            self.chord_vocab = json.load(file)
            
        # inst_vocab_path = '/workspace/data/vocabs/inst.json'
        # with open(inst_vocab_path, 'r') as file:
        #     self.inst_vocab = json.load(file)
            
        # rythm_vocab_path = '/workspace/data/vocabs/rythm.json'
        # with open(rythm_vocab_path, 'r') as file:
        #     self.rythm_vocab = json.load(file)
            
        note_vocab_path = '/workspace/data/vocabs/note.json'
        with open(note_vocab_path, 'r') as file:
            self.note_vocab = json.load(file)
            
        # measure_vocab_path = '/workspace/data/vocabs/measure.json'
        # with open(measure_vocab_path, 'r') as file:
        #     self.measure_vocab = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_seq = self.data[idx]
        
        if isinstance(text_seq, str):
            toks = text_seq.split()
        l_toks = len(toks)
        ratio = 4
        
        measure_list = []
        chord_list = []
        note_list = []
        
        note_in_measure = []
        for idx in range(0, l_toks, ratio):
            # 4개 단위로 txt token 도는중
            t1, t2, t3, t4 = toks[idx : idx + 4]
            
            if t1[0] == 'm' or t1[0] == 'M':
                measure_list.append(t1)
                # 이전 마디 정보 update
                note_list.append(note_in_measure[-self.n_notes:])
                note_in_measure = []
            
            if t1[0] == 'h' or t1[0] == 'H':
                chord_list.append(t1)
                
            if t1 in self.note_vocab:
                if t1 not in note_in_measure:
                    note_in_measure.append(t1)
        # 마무리 note 정보 update
        # 최근 8개
        note_list.append(note_in_measure[-self.n_notes:])
        # 처음 M에서 들어간거 삭제
        note_list = note_list[1:767][:]
        
        # 코드는 Grouping 단위로 해주는 get_chord
        target_chord_seq = self.get_chord_seq(chord_list)
        # 문자에서 숫자로
        target_chord_tensor = [self.chord_vocab[chd] for chd in target_chord_seq]
        target_chord_tensor = [2] + target_chord_tensor[:766] + [1]
        target_chord_tensor = torch.tensor(target_chord_tensor)
        
        # Note 문자에서 숫자로
        target_note_tensor = self.get_note_seq(note_list)
        target_note_tensor = [[2]*self.n_notes] + target_note_tensor + [[1]*self.n_notes]
        target_note_tensor = torch.tensor(target_note_tensor)

        assert target_note_tensor.shape[0] == target_chord_tensor.shape[0]
        
        # RQ
        # if self.base_model == 'RQD':
        #     target_chord_tensor = seq2book(target_chord_tensor)

        return target_chord_tensor, target_note_tensor
    
    # Util functions
    # 3-Group
    # def get_chord_seq(self, chord_list):
    #     group_list = []
    #     for idx in range(2, len(chord_list), 2):
    #         group_list.append(chord_list[idx-2]+chord_list[idx-1]+chord_list[idx])
            
    #     return group_list
    
    # 2-Group
    # def get_chord_seq(self, chord_list):
    #     group_list = []
    #     for idx in range(1, len(chord_list), 2):
    #         group_list.append(chord_list[idx-1]+chord_list[idx])

    #     return group_list
    
    def get_chord_seq(self, chord_list):
        group_list = []
        for idx in range(0, len(chord_list)):
            group_list.append(chord_list[idx])
            
        return group_list
    
    def get_note_seq(self, note_list):
        target_note_tensor = []
        for n_list in note_list:
            n_tensor = [0]*self.n_notes
            if len(n_list) > self.n_notes:
                raise Exception
            
            for idx, n in enumerate(n_list):
                vocab = self.note_vocab[n]
                n_tensor[idx] = vocab
            target_note_tensor.append(n_tensor)
        return target_note_tensor
    
def create_dataloaders(batch_size, base_model, n_notes):
    # chord_data = torch.load('../../../workspace/data/tensor/chord_tensor.pt')
    raw_data_path = '../../../workspace/data/corpus/raw_corpus_bpe.txt'
    raw_data = []
    with open(raw_data_path, 'r') as f:
        for line in tqdm(f, desc="reading original txt file..."):
            raw_data.append(line.strip())
            
    # train, val_test = train_test_split(chord_data, test_size=0.1, random_state=5)
    train, val_test = train_test_split(raw_data, test_size=0.1, random_state=5)
    val, test = train_test_split(val_test, test_size=0.2, random_state=5)
    
    train_dataset = SymDataset(train, base_model, n_notes)
    val_dataset = SymDataset(val, base_model, n_notes)
    test_dataset = SymDataset(test, base_model, n_notes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch)

    return train_loader, val_loader, test_loader

def collate_batch(batch):
    chords, notes = zip(*batch)
    # padding_value = <eos>
    chord_padded = pad_sequence(chords, padding_value=0, batch_first=True)
    note_padded = pad_sequence(notes, padding_value=0, batch_first=True)
    return chord_padded[:,:-1], chord_padded[:,1:], note_padded[:,:-1,:]
    
# RQ
# def collate_batch(batch):
#     # padding_value = <eos>
#     padded = pad_sequence(batch, padding_value=0, batch_first=True)
#     # print("INSIDE BATCH")
#     # print(padded.shape)
#     return padded

# def seq2book(input_ids, book_size=3):
#     # input shape : [seq]
#     # output shape  : [seq+a, codebook]
#     seq_len = input_ids.shape[0]
#     delay = torch.zeros((seq_len, book_size), dtype=torch.long)
#     for i in range(book_size):
#         delay[i:,i] = input_ids[:seq_len-i]
        
#     return delay