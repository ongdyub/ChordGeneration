import torch
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class InstDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        inst_vocab_path = '/workspace/data/vocabs/inst.json'
        chord_vocab_path = '/workspace/data/vocabs/chord.json'
        with open(inst_vocab_path, 'r') as file:
            self.inst_vocab = json.load(file)
        with open(chord_vocab_path, 'r') as file:
            self.chord_vocab = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_seq = self.data[idx]
        
        if isinstance(text_seq, str):
            toks = text_seq.split()
            
        l_toks = len(toks)
        ratio = 4
        chord_list = []
        inst_in_measure = []
        inst_list = []
        
        for idx in range(0, l_toks, ratio):
            t1, t2, t3, t4 = toks[idx : idx + 4]
            if t1[0] == 'H':
                chord_list.append(t1)

            if t4[0] == 'x' or t4[0] == 'X' or t4[0] == 'y' or t4 == '<unk>':
                inst_in_measure.append(t4)
                
            if (t1[0] == 'm' or t1[0] == 'M') and len(chord_list) > 0:
                inst_list.append(inst_in_measure)
                inst_in_measure = []
        inst_list.append(inst_in_measure)
        
        chord_tensor = [self.chord_vocab[chd] for chd in chord_list]
        inst_tensor, length = self.convert_inst_to_onehot(inst_list)
        
        target_chord_tensor = [2] + chord_tensor[:766] + [1]
        target_chord_tensor = torch.tensor(target_chord_tensor)
        
        target_inst_tensor = inst_tensor

        return target_chord_tensor, target_inst_tensor, length+2
    
    def convert_inst_to_onehot(self, inst_list):
        base_tensor = torch.zeros(len(inst_list), 133)
        bos_tensor = torch.zeros(1, 133)
        eos_tensor = torch.zeros(1, 133)
        bos_tensor[:,2] = 1
        eos_tensor[:,1] = 1
        
        for idx, inst_in_measure in enumerate(inst_list):
            if len(inst_in_measure) == 0:
                continue
            else:
                for inst in inst_in_measure:
                    base_tensor[idx, self.inst_vocab[inst]] = 1
        inst_tensor = torch.cat((bos_tensor,base_tensor[:766,:],eos_tensor), dim=0)
        return inst_tensor, len(inst_list)
    
def create_dataloaders(batch_size):
    raw_data_path = '../../../workspace/data/corpus/raw_corpus_bpe.txt'
    # raw_data_path = '../../../workspace/data/corpus/first_5_lines_bpe.txt'
    raw_data = []
    with open(raw_data_path, 'r') as f:
        for line in tqdm(f, desc="reading original txt file..."):
            raw_data.append(line.strip())
            
    train, val_test = train_test_split(raw_data, test_size=0.1, random_state=5)
    val, test = train_test_split(val_test, test_size=0.2, random_state=5)
    # train, val_test = train_test_split(raw_data, test_size=0.5, random_state=5)
    # val, test = train_test_split(val_test, test_size=0.2)
    
    train_dataset = InstDataset(train)
    val_dataset = InstDataset(val)
    test_dataset = InstDataset(test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch)

    # return train_loader, True, True
    return train_loader, val_loader, test_loader

def collate_batch(batch):
    chords, insts, length = zip(*batch)
    # padding_value = <eos>
    chord_padded = pad_sequence(chords, padding_value=0, batch_first=True)
    inst_padded = pad_sequence(insts, padding_value=0, batch_first=True)
    return chord_padded, inst_padded, length



# train_loader, val_loader, test_loader = create_dataloaders(2)

# for chord, inst, length in train_loader:
#     print(chord)
#     print(chord.shape)
#     print(inst.shape)
#     print(inst[:,0])
#     print(inst[:,1])
#     print(inst[:,-1])
#     print(length)
#     break