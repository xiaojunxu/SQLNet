import json
import torch
from sqlnet.utils import *
import numpy as np
import datetime
from collections import Counter

LOCAL_TEST=False


if LOCAL_TEST:
    N_word=100
    B_word=6
    USE_SMALL=True
else:
    N_word=300
    B_word=42
    USE_SMALL=False

char_counter = Counter()

sql_data, table_data, val_sql_data, val_table_data,\
        test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB = \
        load_dataset(0, use_small=USE_SMALL)

char_emb = load_char_emb('glove/glove.840B.300d-char.txt')

word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word),
        use_small=USE_SMALL)


print "Length of word vocabulary: %d"%len(word_emb)

word_to_idx = {'<UNK>':0, '<BEG>':1, '<END>':2}
word_num = 3

word_embs_list = [np.zeros(N_word, dtype=np.float32) for _ in range(word_num)]

char_num = 1
char_to_idx = {' ':0}
char_counter[' '] = 99999
char_embs_list = [np.zeros(N_word,dtype=np.float32) for _ in range(1)]

def check_and_add_out_glove(tok):
    global char_num
    for c in tok:
        if c != ' ' and c not in char_counter and c not in char_emb:
            char_to_idx[c] = char_num
            char_counter[c] += 1
            char_emb[c]=np.random.randn(N_word)
            char_num += 1
            char_embs_list.append(char_emb[c])

    # global word_num
    # if tok not in word_to_idx and tok not in word_emb:
    #     word_to_idx[tok] = word_num
    #     word_num += 1
    #     word_embs_list.append(np.random.randn(N_word))

def check_and_add(tok):
    global char_num
    for c in tok:
        if c != ' ' and c not in char_counter and c in char_emb:
            char_to_idx[c] = char_num
            char_counter[c] += 1
            char_num += 1
            char_embs_list.append(char_emb[c])

    #Check if the tok is in the vocab. If not, add it.
    global word_num
    if tok not in word_to_idx and tok in word_emb:
        word_to_idx[tok] = word_num
        word_num += 1
        word_embs_list.append(word_emb[tok])

for sql in sql_data:
    for tok in sql['question_tok']:
        check_and_add(tok)
for tab in table_data.values():
    for col in tab['header_tok']:
        for tok in col:
            check_and_add(tok)
for sql in val_sql_data:
    for tok in sql['question_tok']:
        check_and_add(tok)
for tab in val_table_data.values():
    for col in tab['header_tok']:
        for tok in col:
            check_and_add(tok)
for sql in test_sql_data:
    for tok in sql['question_tok']:
        check_and_add(tok)
for tab in test_table_data.values():
    for col in tab['header_tok']:
        for tok in col:
            check_and_add(tok)

print "Length of used word vocab: %s"%len(word_to_idx)


for sql in sql_data:
    for tok in sql['question_tok']:
        check_and_add_out_glove(tok)
for tab in table_data.values():
    for col in tab['header_tok']:
        for tok in col:
            check_and_add_out_glove(tok)
for sql in val_sql_data:
    for tok in sql['question_tok']:
        check_and_add_out_glove(tok)
for tab in val_table_data.values():
    for col in tab['header_tok']:
        for tok in col:
            check_and_add_out_glove(tok)
for sql in test_sql_data:
    for tok in sql['question_tok']:
        check_and_add_out_glove(tok)
for tab in test_table_data.values():
    for col in tab['header_tok']:
        for tok in col:
            check_and_add_out_glove(tok)

print "Length of used word vocab: %s"%len(word_to_idx)

temp_length = len(word_to_idx)

emb_array_char = np.stack(char_embs_list, axis=0)
emb_array_word = np.stack(word_embs_list, axis=0)
with open('glove/word2idx.json', 'w') as outf:
    json.dump(word_to_idx, outf)
np.save(open('glove/usedwordemb.npy', 'w'), emb_array_word)
np.save(open('glove/usedcharemb.npy', 'w'), emb_array_char)

char_count_threshold = 0
# char2idx = {char: idx + 1 for idx, char in
#                               enumerate(char for char, count in char_counter.items()
#                                         if count > char_count_threshold)}

final_char_to_idx = {}
for char,count in char_counter.items():
    if count > char_count_threshold:
        final_char_to_idx[char] = char_to_idx[char]

with open('glove/char2idx.json', 'w') as outf:
    json.dump(final_char_to_idx, outf)
