import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm, col_name_encode

class SQLNetCondPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_col_num, max_tok_num, use_ca, gpu):
        super(SQLNetCondPredictor, self).__init__()
        self.N_h = N_h
        self.max_tok_num = max_tok_num
        self.max_col_num = max_col_num
        self.gpu = gpu
        self.use_ca = use_ca

        self.cond_num_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.cond_num_att = nn.Linear(N_h, 1)
        self.cond_num_out = nn.Sequential(nn.Linear(N_h, N_h),
                nn.Tanh(), nn.Linear(N_h, 5))
        self.cond_num_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.cond_num_col_att = nn.Linear(N_h, 1)
        self.cond_num_col2hid1 = nn.Linear(N_h, 2*N_h)
        self.cond_num_col2hid2 = nn.Linear(N_h, 2*N_h)

        self.cond_col_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        if use_ca:
            print "Using column attention on where predicting"
            self.cond_col_att = nn.Linear(N_h, N_h)
        else:
            print "Not using column attention on where predicting"
            self.cond_col_att = nn.Linear(N_h, 1)
        self.cond_col_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.cond_col_out_K = nn.Linear(N_h, N_h)
        self.cond_col_out_col = nn.Linear(N_h, N_h)
        self.cond_col_out = nn.Sequential(nn.ReLU(), nn.Linear(N_h, 1))

        self.cond_op_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        if use_ca:
            self.cond_op_att = nn.Linear(N_h, N_h)
        else:
            self.cond_op_att = nn.Linear(N_h, 1)
        self.cond_op_out_K = nn.Linear(N_h, N_h)
        self.cond_op_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.cond_op_out_col = nn.Linear(N_h, N_h)
        self.cond_op_out = nn.Sequential(nn.Linear(N_h, N_h), nn.Tanh(),
                nn.Linear(N_h, 3))

        self.cond_str_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.cond_str_decoder = nn.LSTM(input_size=self.max_tok_num,
                hidden_size=N_h, num_layers=N_depth,
                batch_first=True, dropout=0.3)
        self.cond_str_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.cond_str_out_g = nn.Linear(N_h, N_h)
        self.cond_str_out_h = nn.Linear(N_h, N_h)
        self.cond_str_out_col = nn.Linear(N_h, N_h)
        self.cond_str_out = nn.Sequential(nn.ReLU(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax()


    def gen_gt_batch(self, split_tok_seq):
        B = len(split_tok_seq)
        max_len = max([max([len(tok) for tok in tok_seq]+[0]) for 
            tok_seq in split_tok_seq]) - 1 # The max seq len in the batch.
        if max_len < 1:
            max_len = 1
        ret_array = np.zeros((
            B, 4, max_len, self.max_tok_num), dtype=np.float32)
        ret_len = np.zeros((B, 4))
        for b, tok_seq in enumerate(split_tok_seq):
            idx = 0
            for idx, one_tok_seq in enumerate(tok_seq):
                out_one_tok_seq = one_tok_seq[:-1]
                ret_len[b, idx] = len(out_one_tok_seq)
                for t, tok_id in enumerate(out_one_tok_seq):
                    ret_array[b, idx, t, tok_id] = 1
            if idx < 3:
                ret_array[b, idx+1:, 0, 1] = 1
                ret_len[b, idx+1:] = 1

        ret_inp = torch.from_numpy(ret_array)
        if self.gpu:
            ret_inp = ret_inp.cuda()
        ret_inp_var = Variable(ret_inp)

        return ret_inp_var, ret_len #[B, IDX, max_len, max_tok_num]


    def forward(self, x_emb_var, x_len, col_inp_var, col_name_len,
            col_len, col_num, gt_where, gt_cond, reinforce):
        max_x_len = max(x_len)
        B = len(x_len)
        if reinforce:
            raise NotImplementedError('Our model doesn\'t have RL')

        # Predict the number of conditions
        # First use column embeddings to calculate the initial hidden unit
        # Then run the LSTM and predict condition number.
        e_num_col, col_num = col_name_encode(col_inp_var, col_name_len,
                col_len, self.cond_num_name_enc)
        num_col_att_val = self.cond_num_col_att(e_num_col).squeeze()
        for idx, num in enumerate(col_num):
            if num < max(col_num):
                num_col_att_val[idx, num:] = -100
        num_col_att = self.softmax(num_col_att_val)
        K_num_col = (e_num_col * num_col_att.unsqueeze(2)).sum(1)
        cond_num_h1 = self.cond_num_col2hid1(K_num_col).view(
                B, 4, self.N_h/2).transpose(0, 1).contiguous()
        cond_num_h2 = self.cond_num_col2hid2(K_num_col).view(
                B, 4, self.N_h/2).transpose(0, 1).contiguous()

        h_num_enc, _ = run_lstm(self.cond_num_lstm, x_emb_var, x_len,
                hidden=(cond_num_h1, cond_num_h2))

        num_att_val = self.cond_num_att(h_num_enc).squeeze()

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                num_att_val[idx, num:] = -100
        num_att = self.softmax(num_att_val)

        K_cond_num = (h_num_enc * num_att.unsqueeze(2).expand_as(
            h_num_enc)).sum(1)
        cond_num_score = self.cond_num_out(K_cond_num)

        #Predict the columns of conditions
        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len, col_len,
                self.cond_col_name_enc)

        h_col_enc, _ = run_lstm(self.cond_col_lstm, x_emb_var, x_len)
        if self.use_ca:
            col_att_val = torch.bmm(e_cond_col,
                    self.cond_col_att(h_col_enc).transpose(1, 2))
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    col_att_val[idx, :, num:] = -100
            col_att = self.softmax(col_att_val.view(
                (-1, max_x_len))).view(B, -1, max_x_len)
            K_cond_col = (h_col_enc.unsqueeze(1) * col_att.unsqueeze(3)).sum(2)
        else:
            col_att_val = self.cond_col_att(h_col_enc).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    col_att_val[idx, num:] = -100
            col_att = self.softmax(col_att_val)
            K_cond_col = (h_col_enc *
                    col_att_val.unsqueeze(2)).sum(1).unsqueeze(1)

        cond_col_score = self.cond_col_out(self.cond_col_out_K(K_cond_col) +
                self.cond_col_out_col(e_cond_col)).squeeze()
        max_col_num = max(col_num)
        for b, num in enumerate(col_num):
            if num < max_col_num:
                cond_col_score[b, num:] = -100

        #Predict the operator of conditions
        chosen_col_gt = []
        if gt_cond is None:
            cond_nums = np.argmax(cond_num_score.data.cpu().numpy(), axis=1)
            col_scores = cond_col_score.data.cpu().numpy()
            chosen_col_gt = [list(np.argsort(-col_scores[b])[:cond_nums[b]])
                    for b in range(len(cond_nums))]
        else:
            chosen_col_gt = [ [x[0] for x in one_gt_cond] for 
                    one_gt_cond in gt_cond]

        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len,
                col_len, self.cond_op_name_enc)
        col_emb = []
        for b in range(B):
            cur_col_emb = torch.stack([e_cond_col[b, x] 
                for x in chosen_col_gt[b]] + [e_cond_col[b, 0]] *
                (4 - len(chosen_col_gt[b])))  # Pad the columns to maximum (4)
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb)

        h_op_enc, _ = run_lstm(self.cond_op_lstm, x_emb_var, x_len)
        if self.use_ca:
            op_att_val = torch.matmul(self.cond_op_att(h_op_enc).unsqueeze(1),
                    col_emb.unsqueeze(3)).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    op_att_val[idx, :, num:] = -100
            op_att = self.softmax(op_att_val.view(B*4, -1)).view(B, 4, -1)
            K_cond_op = (h_op_enc.unsqueeze(1) * op_att.unsqueeze(3)).sum(2)
        else:
            op_att_val = self.cond_op_att(h_op_enc).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    op_att_val[idx, num:] = -100
            op_att = self.softmax(op_att_val)
            K_cond_op = (h_op_enc * op_att.unsqueeze(2)).sum(1).unsqueeze(1)

        cond_op_score = self.cond_op_out(self.cond_op_out_K(K_cond_op) +
                self.cond_op_out_col(col_emb)).squeeze()

        #Predict the string of conditions
        h_str_enc, _ = run_lstm(self.cond_str_lstm, x_emb_var, x_len)
        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len,
                col_len, self.cond_str_name_enc)
        col_emb = []
        for b in range(B):
            cur_col_emb = torch.stack([e_cond_col[b, x]
                for x in chosen_col_gt[b]] +
                [e_cond_col[b, 0]] * (4 - len(chosen_col_gt[b])))
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb)

        if gt_where is not None:
            gt_tok_seq, gt_tok_len = self.gen_gt_batch(gt_where)
            g_str_s_flat, _ = self.cond_str_decoder(
                    gt_tok_seq.view(B*4, -1, self.max_tok_num))
            g_str_s = g_str_s_flat.contiguous().view(B, 4, -1, self.N_h)

            h_ext = h_str_enc.unsqueeze(1).unsqueeze(1)
            g_ext = g_str_s.unsqueeze(3)
            col_ext = col_emb.unsqueeze(2).unsqueeze(2)

            cond_str_score = self.cond_str_out(
                    self.cond_str_out_h(h_ext) + self.cond_str_out_g(g_ext) +
                    self.cond_str_out_col(col_ext)).squeeze()
            for b, num in enumerate(x_len):
                if num < max_x_len:
                    cond_str_score[b, :, :, num:] = -100
        else:
            h_ext = h_str_enc.unsqueeze(1).unsqueeze(1)
            col_ext = col_emb.unsqueeze(2).unsqueeze(2)
            scores = []

            t = 0
            init_inp = np.zeros((B*4, 1, self.max_tok_num), dtype=np.float32)
            init_inp[:,0,0] = 1  #Set the <BEG> token
            if self.gpu:
                cur_inp = Variable(torch.from_numpy(init_inp).cuda())
            else:
                cur_inp = Variable(torch.from_numpy(init_inp))
            cur_h = None
            while t < 50:
                if cur_h:
                    g_str_s_flat, cur_h = self.cond_str_decoder(cur_inp, cur_h)
                else:
                    g_str_s_flat, cur_h = self.cond_str_decoder(cur_inp)
                g_str_s = g_str_s_flat.view(B, 4, 1, self.N_h)
                g_ext = g_str_s.unsqueeze(3)

                cur_cond_str_score = self.cond_str_out(
                        self.cond_str_out_h(h_ext) + self.cond_str_out_g(g_ext)
                        + self.cond_str_out_col(col_ext)).squeeze()
                for b, num in enumerate(x_len):
                    if num < max_x_len:
                        cur_cond_str_score[b, :, num:] = -100
                scores.append(cur_cond_str_score)

                _, ans_tok_var = cur_cond_str_score.view(B*4, max_x_len).max(1)
                ans_tok = ans_tok_var.data.cpu()
                data = torch.zeros(B*4, self.max_tok_num).scatter_(
                        1, ans_tok.unsqueeze(1), 1)
                if self.gpu:  #To one-hot
                    cur_inp = Variable(data.cuda())
                else:
                    cur_inp = Variable(data)
                cur_inp = cur_inp.unsqueeze(1)

                t += 1

            cond_str_score = torch.stack(scores, 2)
            for b, num in enumerate(x_len):
                if num < max_x_len:
                    cond_str_score[b, :, :, num:] = -100  #[B, IDX, T, TOK_NUM]

        cond_score = (cond_num_score,
                cond_col_score, cond_op_score, cond_str_score)

        return cond_score
