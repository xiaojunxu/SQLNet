import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class WordEmbedding(nn.Module):
    def __init__(self, word_emb,char_emb, N_word, gpu, SQL_TOK,
            our_model, trainable=False):
        super(WordEmbedding, self).__init__()
        self.trainable = trainable
        self.N_word = N_word
        self.our_model = our_model
        self.gpu = gpu
        self.SQL_TOK = SQL_TOK
        self.char_emb = char_emb

        if trainable:
              print "Using trainable embedding"
              self.w2i, word_emb_val = word_emb
              self.embedding = nn.Embedding(len(self.w2i), N_word)
              self.embedding.weight = nn.Parameter(
                    torch.from_numpy(word_emb_val.astype(np.float32)))
        else:
            self.word_emb = word_emb
            print "Using fixed embedding"

        if char_emb != None:
            self.c2i, char_emb_val = char_emb

            self.char_embedding = nn.Embedding(len(self.c2i), N_word)
            self.char_embedding.weight = nn.Parameter(
                    torch.from_numpy(char_emb_val.astype(np.float32)))

            in_channels = 300
            out_channels = 100
            self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 5))
            self.conv2 = nn.Conv2d(in_channels, out_channels, (1, 4))
            self.conv3 = nn.Conv2d(in_channels, out_channels, (1, 3))
            self.dropout = nn.Dropout(0.3)

    def get_question_char(self,q):
        result = []
        max_char_len = 0
        for one_question in q:
            one_question_result = []
            for word in one_question:
                one_word_result = []
                count = 0
                for c in word:
                    count+=1
                    if count>max_char_len:
                        max_char_len = count
                    # if c != ' ':
                    one_word_result.append(self.c2i[c])
                one_question_result.append(one_word_result)
            result.append(one_question_result)
        return result,max_char_len

    def get_col_char2(self,cols):
        result = []
        for one_col_list in cols:
            one_col_list_result = []
            for one_col in one_col_list:
                one_col_result = []
                for word in one_col:
                    one_word_result = []
                    for c in word:
                        # if c != ' ':
                        one_word_result.append(self.c2i[c])
                    one_col_result.append(one_word_result)
                one_col_list_result.append(one_col_result)
            result.append(one_col_list_result)
        return result

    def get_col_char(self,cols):
        result = []
        max_char_len = 0
        for one_list in cols:
            one_list_result = []
            for word in one_list:
                one_word_result = []
                count = 0
                for c in word:
                    count += 1
                    if count > max_char_len:
                        max_char_len = count
                        # if c != ' ':
                    one_word_result.append(self.c2i[c])
                one_list_result.append(one_word_result)
            result.append(one_list_result)
        return result,max_char_len

    def gen_x_batch(self, q, col):
        #WORD
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, (one_q, one_col) in enumerate(zip(q, col)):
            if self.trainable:
                q_val = map(lambda x:self.w2i.get(x, 0), one_q)
            else:
                q_val = map(lambda x:self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), one_q)


            if self.our_model:
                if self.trainable:
                    val_embs.append([1] + q_val + [2])  #<BEG> and <END>
                else:
                    val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + q_val + [np.zeros(self.N_word, dtype=np.float32)])  #<BEG> and <END>
                val_len[i] = 1 + len(q_val) + 1
            else:
                one_col_all = [x for toks in one_col for x in toks+[',']]
                if self.trainable:
                    col_val = map(lambda x:self.w2i.get(x, 0), one_col_all)
                    val_embs.append( [0 for _ in self.SQL_TOK] + col_val + [0] + q_val+ [0])
                else:
                    col_val = map(lambda x:self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), one_col_all)
                    val_embs.append( [np.zeros(self.N_word, dtype=np.float32) for _ in self.SQL_TOK] + col_val + [np.zeros(self.N_word, dtype=np.float32)] + q_val+ [np.zeros(self.N_word, dtype=np.float32)])
                val_len[i] = len(self.SQL_TOK) + len(col_val) + 1 + len(q_val) + 1
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)

        # CHAR
        if self.char_emb != None:
            B = len(q)
            q_char,max_char_len = self.get_question_char(q)
            col_char = self.get_col_char2(col)

            temp = np.zeros((B, max_len, max_char_len), dtype=np.int64)
            for i in range(B):
                for j in range(len(q_char[i])):
                    for k in range(len(q_char[i][j])):
                        temp[i, j, k] = q_char[i][j][k]
            temp2 = torch.from_numpy(temp)
            if self.gpu:
                temp2 = temp2.cuda()
            temp3 = Variable(temp2)
            temp3 = temp3.view(B,-1)
            char_emb_temp = self.char_embedding(temp3)
            char_emb_temp = char_emb_temp.view(B, max_len, max_char_len,self.N_word).permute(0,3,1,2)

            # N_word -- in_channel
            # conv input: N C_in H W
            # conv ouput: N C_out H_ W_

            char_result1 = F.relu(self.conv1(char_emb_temp))
            char_result2 = F.relu(self.conv2(char_emb_temp))
            char_result3 = F.relu(self.conv3(char_emb_temp))

            char_result1_ = char_result1.max(3)[0].permute(0,2,1)
            char_result2_ = char_result2.max(3)[0].permute(0,2,1)
            char_result3_ = char_result3.max(3)[0].permute(0,2,1)

            char_result_final = torch.cat([char_result1_,char_result2_,char_result3_],-1)

            char_result_final = self.dropout(char_result_final)

            val_inp_var = torch.cat([char_result_final,val_inp_var],-1)

        return val_inp_var, val_len

    def gen_col_batch(self, cols):
        ret = []
        col_len = np.zeros(len(cols), dtype=np.int64)

        names = []
        for b, one_cols in enumerate(cols):
            names = names + one_cols
            col_len[b] = len(one_cols)

        name_inp_var, name_len = self.str_list_to_batch(names)
        return name_inp_var, name_len, col_len

    def str_list_to_batch(self, str_list):
        B = len(str_list)
        # WORD
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_str in enumerate(str_list):
            if self.trainable:
                val = [self.w2i.get(x, 0) for x in one_str]
            else:
                val = [self.word_emb.get(x, np.zeros(
                    self.N_word, dtype=np.float32)) for x in one_str]
            val_embs.append(val)
            val_len[i] = len(val)
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros(
                    (B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)

        #CHAR
        if self.char_emb != None:
            col_char,max_char_len = self.get_col_char(str_list)

            temp = np.zeros((B, max_len, max_char_len), dtype=np.int64)
            for i in range(B):
                for j in range(len(col_char[i])):
                    for k in range(len(col_char[i][j])):
                        temp[i, j, k] = col_char[i][j][k]
            temp2 = torch.from_numpy(temp)
            if self.gpu:
                temp2 = temp2.cuda()
            temp3 = Variable(temp2)
            temp3 = temp3.view(B, -1)
            char_emb_temp = self.char_embedding(temp3)
            char_emb_temp = char_emb_temp.view(B, max_len, max_char_len, self.N_word).permute(0, 3, 1, 2)

            # N_word -- in_channel
            # conv input: N C_in H W
            # conv ouput: N C_out H_ W_

            char_result1 = F.relu(self.conv1(char_emb_temp))
            char_result2 = F.relu(self.conv2(char_emb_temp))
            char_result3 = F.relu(self.conv3(char_emb_temp))

            char_result1_ = char_result1.max(3)[0].permute(0, 2, 1)
            char_result2_ = char_result2.max(3)[0].permute(0, 2, 1)
            char_result3_ = char_result3.max(3)[0].permute(0, 2, 1)

            char_result_final = torch.cat([char_result1_, char_result2_, char_result3_], -1)

            char_result_final = self.dropout(char_result_final)

            val_inp_var = torch.cat([char_result_final, val_inp_var], -1)

        return val_inp_var, val_len
