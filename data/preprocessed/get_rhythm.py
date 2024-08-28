import time, os, json
from collections import Counter
from pprint import pprint
from tqdm import tqdm
import subprocess#, multiprocessing
from functools import partial
from p_tqdm import p_uimap
RATIO = 4
MERGE_CNT = 700
CHAR_CNT = 128
WORKERS = 32

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encoding import pit2str, str2pit, ispitch

def count_rhythm_token(toks, ratio=RATIO):
    # 1개의 곡에 대해서 수행하는 함수
    if isinstance(toks, str):
        toks = toks.split()

    mulpies = dict()
    chord_dict = Counter()
    l_toks = len(toks)
    prefix = ''
    for idx in range(0, l_toks, ratio):
        t1, t2, t3, t4 = toks[idx : idx + 4]
        if t1[0] == 'm' or t1[0] == 'M':
            if prefix == '':
                continue
            chord_dict[prefix] += 1
            prefix = ''
        
        if t1 == 'NT':
            if prefix == '':
                continue
            chord_dict[prefix] += 1
            prefix = ''
            
        if t1[0] == 'p' or t1[0] == 'P':
            if prefix == '':
                prefix = t1
                continue
            chord_dict[prefix] += 1
            prefix = t1
            
        if ispitch(t1[0:2]):
            prefix += t2

    return chord_dict, l_toks // ratio


if __name__ == '__main__':
    start_time = time.time()

    paragraphs = []

    raw_data_path = '../../data/corpus/raw_corpus_bpe.txt'
    output_dir = '../../data/corpus/'
    os.makedirs(output_dir, exist_ok=True)
    raw_data = []
    with open(raw_data_path, 'r') as f:
        for line in tqdm(f, desc="reading original txt file..."):
            raw_data.append(line.strip())

    chord_dict = Counter()
    before_total_tokens = 0
    for sub_chord_dict, l_toks in p_uimap(count_rhythm_token, raw_data, num_cpus=WORKERS):
        chord_dict += sub_chord_dict
        before_total_tokens += l_toks
    
    mulpi_list = sorted(chord_dict.most_common(), key=lambda x: (-x[1], x[0]))
    with open(output_dir+'rhythm_token_cnt.txt', 'w') as f:
        f.write(str(len(mulpi_list)) + '\n')
        for k, v in mulpi_list:
            f.write(''.join(k) + ' ' + str(v) + '\n')
