import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm

class Seq2SQLCondPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_col_num, max_tok_num, gpu):
        super(Seq2SQLCondPredictor, self).__init__()
        print "Seq2SQL where prediction"
        self.N_h = N_h
        self.max_tok_num = max_tok_num
        self.max_col_num = max_col_num
        self.gpu = gpu

        self.cond_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.cond_decoder = nn.LSTM(input_size=self.max_tok_num,
                hidden_size=N_h, num_layers=N_depth,
                batch_first=True, dropout=0.3)

        self.cond_out_g = nn.Linear(N_h, N_h)
        self.cond_out_h = nn.Linear(N_h, N_h)
        self.cond_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax()


    def gen_gt_batch(self, tok_seq, gen_inp=True):
        # If gen_inp: generate the input token sequence (removing <END>)
        # Otherwise: generate the output token sequence (removing <BEG>)
        B = len(tok_seq)
        ret_len = np.array([len(one_tok_seq)-1 for one_tok_seq in tok_seq])
        max_len = max(ret_len)
        ret_array = np.zeros((B, max_len, self.max_tok_num), dtype=np.float32)
        for b, one_tok_seq in enumerate(tok_seq):
            out_one_tok_seq = one_tok_seq[:-1] if gen_inp else one_tok_seq[1:]
            for t, tok_id in enumerate(out_one_tok_seq):
                ret_array[b, t, tok_id] = 1

        ret_inp = torch.from_numpy(ret_array)
        if self.gpu:
            ret_inp = ret_inp.cuda()
        ret_inp_var = Variable(ret_inp) #[B, max_len, max_tok_num]

        return ret_inp_var, ret_len


    def forward(self, x_emb_var, x_len, col_inp_var, col_name_len, col_len,
            col_num, gt_where, gt_cond, reinforce):
        max_x_len = max(x_len)
        B = len(x_len)

        h_enc, hidden = run_lstm(self.cond_lstm, x_emb_var, x_len)
        decoder_hidden = tuple(torch.cat((hid[:2], hid[2:]),dim=2) 
                for hid in hidden)
        if gt_where is not None:
            gt_tok_seq, gt_tok_len = self.gen_gt_batch(gt_where, gen_inp=True)
            g_s, _ = run_lstm(self.cond_decoder,
                    gt_tok_seq, gt_tok_len, decoder_hidden)

            h_enc_expand = h_enc.unsqueeze(1)
            g_s_expand = g_s.unsqueeze(2)
            cond_score = self.cond_out( self.cond_out_h(h_enc_expand) +
                    self.cond_out_g(g_s_expand) ).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    cond_score[idx, :, num:] = -100
        else:
            h_enc_expand = h_enc.unsqueeze(1)
            scores = []
            choices = []
            done_set = set()

            t = 0
            init_inp = np.zeros((B, 1, self.max_tok_num), dtype=np.float32)
            init_inp[:,0,7] = 1  #Set the <BEG> token
            if self.gpu:
                cur_inp = Variable(torch.from_numpy(init_inp).cuda())
            else:
                cur_inp = Variable(torch.from_numpy(init_inp))
            cur_h = decoder_hidden
            while len(done_set) < B and t < 100:
                g_s, cur_h = self.cond_decoder(cur_inp, cur_h)
                g_s_expand = g_s.unsqueeze(2)

                cur_cond_score = self.cond_out(self.cond_out_h(h_enc_expand) +
                        self.cond_out_g(g_s_expand)).squeeze()
                for b, num in enumerate(x_len):
                    if num < max_x_len:
                        cur_cond_score[b, num:] = -100
                scores.append(cur_cond_score)

                if not reinforce:
                    _, ans_tok_var = cur_cond_score.view(B, max_x_len).max(1)
                    ans_tok_var = ans_tok_var.unsqueeze(1)
                else:
                    ans_tok_var = self.softmax(cur_cond_score).multinomial()
                    choices.append(ans_tok_var)
                ans_tok = ans_tok_var.data.cpu()
                if self.gpu:  #To one-hot
                    cur_inp = Variable(torch.zeros(
                        B, self.max_tok_num).scatter_(1, ans_tok, 1).cuda())
                else:
                    cur_inp = Variable(torch.zeros(
                        B, self.max_tok_num).scatter_(1, ans_tok, 1))
                cur_inp = cur_inp.unsqueeze(1)

                for idx, tok in enumerate(ans_tok.squeeze()):
                    if tok == 1:  #Find the <END> token
                        done_set.add(idx)
                t += 1

            cond_score = torch.stack(scores, 1)

        if reinforce:
            return cond_score, choices
        else:
            return cond_score
