import sys

from tqdm import tqdm
import json
import torch

raw_data_path = '/workspace/pj/data/corpus/raw_corpus_bpe.txt'
# raw_data_path = '/workspace/data/corpus/first_5_lines_bpe.txt'

vocab_path = '/workspace/pj/data/vocabs/chord.json'
with open(vocab_path, 'r') as file:
    vocab = json.load(file)

# print("Initial Vocabulary:", vocab)


raw_data = []
with open(raw_data_path, 'r') as f:
    for line in tqdm(f, desc="reading original txt file..."):
        raw_data.append(line.strip())


def extract_chord_seq_list(toks, ratio=4):
    if isinstance(toks, str):
        toks = toks.split()

    l_toks = len(toks)

    chord_list = []
    
    for idx in range(0, l_toks, ratio):
        t1, t2, t3, t4 = toks[idx : idx + 4]
        
        if t1[0] == 'h' or t1[0] == 'H':
            chord_list.append(t1)
            
    return chord_list

# chord sequnece만 추출

chord_seq = []
cnt = 0
for data in tqdm(raw_data):
    out_list = extract_chord_seq_list(data)
    chord_seq.append(out_list)
    cnt += 1
    
print("TOTAL CHORD SEQ LENGTH")
print(len(chord_seq))
# print(chord_seq[0])

init_dict = {"<eos>": 1,
    "<pad>": 0,
    "<bos>": 2,
    "HC+": 134,
    "HC/o7": 135,
    "HCD7": 3,
    "HCM": 4,
    "HCM7": 5,
    "HCm": 6,
    "HCm7": 7,
    "HCo": 8,
    "HCo7": 9,
    "HCsus2": 10,
    "HCsus4": 11,
    "Hd+": 12,
    "Hd/o7": 13,
    "HdD7": 14,
    "HdM": 15,
    "HdM7": 16,
    "Hdm": 17,
    "Hdm7": 18,
    "Hdo": 19,
    "Hdo7": 20,
    "Hdsus2": 21,
    "Hdsus4": 22,
    "HD+": 23,
    "HD/o7": 24,
    "HDD7": 25,
    "HDM": 26,
    "HDM7": 27,
    "HDm": 28,
    "HDm7": 29,
    "HDo": 30,
    "HDo7": 31,
    "HDsus2": 32,
    "HDsus4": 33,
    "He+": 34,
    "He/o7": 35,
    "HeD7": 36,
    "HeM": 37,
    "HeM7": 38,
    "Hem": 39,
    "Hem7": 40,
    "Heo": 41,
    "Heo7": 42,
    "Hesus2": 43,
    "Hesus4": 44,
    "HE+": 45,
    "HE/o7": 46,
    "HED7": 47,
    "HEM": 48,
    "HEM7": 49,
    "HEm": 50,
    "HEm7": 51,
    "HEo": 52,
    "HEo7": 53,
    "HEsus2": 54,
    "HEsus4": 55,
    "HF+": 56,
    "HF/o7": 57,
    "HFD7": 58,
    "HFM": 59,
    "HFM7": 60,
    "HFm": 61,
    "HFm7": 62,
    "HFo": 63,
    "HFo7": 64,
    "HFsus2": 65,
    "HFsus4": 66,
    "Hg+": 67,
    "Hg/o7": 68,
    "HgD7": 69,
    "HgM": 70,
    "HgM7": 71,
    "Hgm": 72,
    "Hgm7": 73,
    "Hgo": 74,
    "Hgo7": 75,
    "Hgsus2": 76,
    "Hgsus4": 77,
    "HG+": 78,
    "HG/o7": 79,
    "HGD7": 80,
    "HGM": 81,
    "HGM7": 82,
    "HGm": 83,
    "HGm7": 84,
    "HGo": 85,
    "HGo7": 86,
    "HGsus2": 87,
    "HGsus4": 88,
    "Ha+": 89,
    "Ha/o7": 90,
    "HaD7": 91,
    "HaM": 92,
    "HaM7": 93,
    "Ham": 94,
    "Ham7": 95,
    "Hao": 96,
    "Hao7": 97,
    "Hasus2": 98,
    "Hasus4": 99,
    "HA+": 100,
    "HA/o7": 101,
    "HAD7": 102,
    "HAM": 103,
    "HAM7": 104,
    "HAm": 105,
    "HAm7": 106,
    "HAo": 107,
    "HAo7": 108,
    "HAsus2": 109,
    "HAsus4": 110,
    "Hb+": 111,
    "Hb/o7": 112,
    "HbD7": 113,
    "HbM": 114,
    "HbM7": 115,
    "Hbm": 116,
    "Hbm7": 117,
    "Hbo": 118,
    "Hbo7": 119,
    "Hbsus2": 120,
    "Hbsus4": 121,
    "HB+": 122,
    "HB/o7": 123,
    "HBD7": 124,
    "HBM": 125,
    "HBM7": 126,
    "HBm": 127,
    "HBm7": 128,
    "HBo": 129,
    "HBo7": 130,
    "HBsus2": 131,
    "HBsus4": 132,
    "HNA": 133
}

for i in init_dict.keys():
    init_dict[i] = 0

print("Init Dict")
# print(init_dict)

init_vocabs = list(init_dict.keys())
print("Init Vocabs")
# print(init_vocabs)
print(len(init_vocabs))

NUM_ITER = 20000

for _ in tqdm(range(NUM_ITER), ncols=60):
    bpe_memory = {}
    for c_seq in chord_seq:
        for idx in range(1,len(c_seq)):
            adj_chord = c_seq[idx-1] + c_seq[idx]
            if adj_chord in bpe_memory:
                bpe_memory[adj_chord] += 1
            else:
                bpe_memory[adj_chord] = 1
    bpe_memory = dict(sorted(bpe_memory.items(), key=lambda item: item[1], reverse=True))
    # print(bpe_memory)
    # print(list(bpe_memory.keys())[0])

    new_vocab = list(bpe_memory.keys())[0]

    if new_vocab in init_vocabs:
        print("ERROR EXIST")
    else:
        init_vocabs.append(new_vocab)

    print("NEW Vocab INFO")
    print(f'ADD [[[{new_vocab}]]] in Dict')
    # print(init_vocabs)
    print(len(init_vocabs))
   
    # merge exist sequence with new vocab
    update_data = []
    for c_seq in chord_seq:
        update_chord = []
        idx = 1
        while(idx < len(c_seq)):
            before = c_seq[idx-1]
            current = c_seq[idx]
            
            if before + current == new_vocab:
                # print("MERGED")
                update_chord.append(before+current)
                idx += 2
            else:
                update_chord.append(before)
                idx += 1
                
            if idx == len(c_seq):
                update_chord.append(c_seq[idx-1])
                break
            if idx == len(c_seq)+1:
                break
        update_data.append(update_chord)
        
    chord_seq = update_data
    
    out_path = f'/workspace/pj/exp/bpe_chord_vocab_{NUM_ITER}.json'
    with open(out_path, 'w') as f:
        json.dump(init_vocabs, f)