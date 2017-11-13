import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from modules.word_embedding import WordEmbedding
from modules.aggregator_predict import AggPredictor
from modules.selection_predict import SelPredictor
from modules.seq2sql_condition_predict import Seq2SQLCondPredictor

# This is a re-implementation based on the following paper:

# Victor Zhong, Caiming Xiong, and Richard Socher. 2017.
# Seq2SQL: Generating Structured Queries from Natural Language using
# Reinforcement Learning. arXiv:1709.00103

class Seq2SQL(nn.Module):
    def __init__(self, word_emb, N_word, N_h=100, N_depth=2,
                 gpu=False, trainable_emb=False):
        super(Seq2SQL, self).__init__()
        self.trainable_emb = trainable_emb

        self.gpu = gpu
        self.N_h = N_h
        self.N_depth = N_depth

        self.max_col_num = 45
        self.max_tok_num = 200
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND',
                        'EQL', 'GT', 'LT', '<BEG>']
        self.COND_OPS = ['EQL', 'GT', 'LT']

        #Word embedding
        if trainable_emb:
            self.agg_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                                                 self.SQL_TOK, our_model=False,
                                                 trainable=trainable_emb)
            self.sel_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                                                 self.SQL_TOK, our_model=False,
                                                 trainable=trainable_emb)
            self.cond_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                                                  self.SQL_TOK, our_model=False,
                                                  trainable=trainable_emb)
        else:
            self.embed_layer = WordEmbedding(word_emb, N_word, gpu,
                                             self.SQL_TOK, our_model=False,
                                             trainable=trainable_emb)

        #Predict aggregator
        self.agg_pred = AggPredictor(N_word, N_h, N_depth, use_ca=False)

        #Predict selected column
        self.sel_pred = SelPredictor(N_word, N_h, N_depth, self.max_tok_num,
                                     use_ca=False)

        #Predict number of cond
        self.cond_pred = Seq2SQLCondPredictor(
            N_word, N_h, N_depth, self.max_col_num, self.max_tok_num, gpu)


        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.bce_logit = nn.BCEWithLogitsLoss()
        if gpu:
            self.cuda()


    def generate_gt_where_seq(self, q, col, query):
        # data format
        # <BEG> WHERE cond1_col cond1_op cond1
        #         AND cond2_col cond2_op cond2
        #         AND ... <END>

        ret_seq = []
        for cur_q, cur_col, cur_query in zip(q, col, query):
            connect_col = [tok for col_tok in cur_col for tok in col_tok+[',']]
            all_toks = self.SQL_TOK + connect_col + [None] + cur_q + [None]
            cur_seq = [all_toks.index('<BEG>')]
            if 'WHERE' in cur_query:
                cur_where_query = cur_query[cur_query.index('WHERE'):]
                cur_seq = cur_seq + map(lambda tok:all_toks.index(tok)
                                        if tok in all_toks else 0, cur_where_query)
            cur_seq.append(all_toks.index('<END>'))
            ret_seq.append(cur_seq)
        return ret_seq


    def forward(self, q, col, col_num, pred_entry,
                gt_where = None, gt_cond=None, reinforce=False, gt_sel=None):
        B = len(q)
        pred_agg, pred_sel, pred_cond = pred_entry

        agg_score = None
        sel_score = None
        cond_score = None

        if self.trainable_emb:
            if pred_agg:
                x_emb_var, x_len = self.agg_embed_layer.gen_x_batch(q, col)
                batch = self.agg_embed_layer.gen_col_batch(col)
                col_inp_var, col_name_len, col_len = batch
                max_x_len = max(x_len)
                agg_score = self.agg_pred(x_emb_var, x_len)

            if pred_sel:
                x_emb_var, x_len = self.sel_embed_layer.gen_x_batch(q, col)
                batch = self.sel_embed_layer.gen_col_batch(col)
                col_inp_var, col_name_len, col_len = batch
                max_x_len = max(x_len)
                sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var,
                                          col_name_len, col_len, col_num)

            if pred_cond:
                x_emb_var, x_len = self.cond_embed_layer.gen_x_batch(q, col)
                batch = self.cond_embed_layer.gen_col_batch(col)
                col_inp_var, col_name_len, col_len = batch
                max_x_len = max(x_len)
                cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var,
                                            col_name_len, col_len, col_num,
                                            gt_where, gt_cond,
                                            reinforce=reinforce)
        else:
            x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col)
            batch = self.embed_layer.gen_col_batch(col)
            col_inp_var, col_name_len, col_len = batch
            max_x_len = max(x_len)
            if pred_agg:
                agg_score = self.agg_pred(x_emb_var, x_len)

            if pred_sel:
                sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var,
                                          col_name_len, col_len, col_num)

            if pred_cond:
                cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var,
                                            col_name_len, col_len, col_num,
                                            gt_where, gt_cond,
                                            reinforce=reinforce)

        return (agg_score, sel_score, cond_score)

    def loss(self, score, truth_num, pred_entry, gt_where):
        pred_agg, pred_sel, pred_cond = pred_entry
        agg_score, sel_score, cond_score = score
        loss = 0
        if pred_agg:
            agg_truth = map(lambda x:x[0], truth_num)
            data = torch.from_numpy(np.array(agg_truth))
            if self.gpu:
                agg_truth_var = Variable(data.cuda())
            else:
                agg_truth_var = Variable(data)

            loss += self.CE(agg_score, agg_truth_var)

        if pred_sel:
            sel_truth = map(lambda x:x[1], truth_num)
            data = torch.from_numpy(np.array(sel_truth))
            if self.gpu:
                sel_truth_var = Variable(data).cuda()
            else:
                sel_truth_var = Variable(data)

            loss += self.CE(sel_score, sel_truth_var)

        if pred_cond:
            for b in range(len(gt_where)):
                if self.gpu:
                    cond_truth_var = Variable(
                        torch.from_numpy(np.array(gt_where[b][1:])).cuda())
                else:
                    cond_truth_var = Variable(
                        torch.from_numpy(np.array(gt_where[b][1:])))
                cond_pred_score = cond_score[b, :len(gt_where[b])-1]

                loss += ( self.CE(
                    cond_pred_score, cond_truth_var) / len(gt_where) )

        return loss

    def reinforce_backward(self, score, rewards):
        agg_score, sel_score, cond_score = score

        cur_reward = rewards[:]
        eof = self.SQL_TOK.index('<END>')
        for t in range(len(cond_score[1])):
            reward_inp = torch.FloatTensor(cur_reward).unsqueeze(1)
            if self.gpu:
                reward_inp = reward_inp.cuda()
            cond_score[1][t].reinforce(reward_inp)

            for b in range(len(rewards)):
                if cond_score[1][t][b].data.cpu().numpy()[0] == eof:
                    cur_reward[b] = 0
        torch.autograd.backward(cond_score[1], [None for _ in cond_score[1]])
        return

    def check_acc(self, vis_info, pred_queries, gt_queries, pred_entry):
        def pretty_print(vis_data):
            print 'question:', vis_data[0]
            print 'headers: (%s)'%(' || '.join(vis_data[1]))
            print 'query:', vis_data[2]

        def gen_cond_str(conds, header):
            if len(conds) == 0:
                return 'None'
            cond_str = []
            for cond in conds:
                cond_str.append(
                    header[cond[0]] + ' ' + self.COND_OPS[cond[1]] + \
                    ' ' + unicode(cond[2]).lower())
            return 'WHERE ' + ' AND '.join(cond_str)

        pred_agg, pred_sel, pred_cond = pred_entry

        B = len(gt_queries)

        tot_err = agg_err = sel_err = cond_err = cond_num_err = \
                  cond_col_err = cond_op_err = cond_val_err = 0.0
        agg_ops = ['None', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
            good = True
            if pred_agg:
                agg_pred = pred_qry['agg']
                agg_gt = gt_qry['agg']
                if agg_pred != agg_gt:
                    agg_err += 1
                    good = False

            if pred_sel:
                sel_pred = pred_qry['sel']
                sel_gt = gt_qry['sel']
                if sel_pred != sel_gt:
                    sel_err += 1
                    good = False

            if pred_cond:
                cond_pred = pred_qry['conds']
                cond_gt = gt_qry['conds']
                flag = True
                if len(cond_pred) != len(cond_gt):
                    flag = False
                    cond_num_err += 1

                if flag and set(
                        x[0] for x in cond_pred) != set(x[0] for x in cond_gt):
                    flag = False
                    cond_col_err += 1

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and cond_gt[gt_idx][1] != cond_pred[idx][1]:
                        flag = False
                        cond_op_err += 1

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and unicode(cond_gt[gt_idx][2]).lower() != \
                       unicode(cond_pred[idx][2]).lower():
                        flag = False
                        cond_val_err += 1

                if not flag:
                    cond_err += 1
                    good = False

            if not good:
                tot_err += 1

        return np.array((agg_err, sel_err, cond_err)), tot_err


    def gen_query(self, score, q, col, raw_q, raw_col, pred_entry,
                  reinforce=False, verbose=False):
        def merge_tokens(tok_list, raw_tok_str):
            tok_str = raw_tok_str.lower()
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
            special = {'-LRB-':'(', '-RRB-':')', '-LSB-':'[', '-RSB-':']',
                       '``':'"', '\'\'':'"', '--':u'\u2013'}
            ret = ''
            double_quote_appear = 0
            for raw_tok in tok_list:
                if not raw_tok:
                    continue
                tok = special.get(raw_tok, raw_tok)
                if tok == '"':
                    double_quote_appear = 1 - double_quote_appear

                if len(ret) == 0:
                    pass
                elif len(ret) > 0 and ret + ' ' + tok in tok_str:
                    ret = ret + ' '
                elif len(ret) > 0 and ret + tok in tok_str:
                    pass
                elif tok == '"':
                    if double_quote_appear:
                        ret = ret + ' '
                elif tok[0] not in alphabet:
                    pass
                elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) and \
                     (ret[-1] != '"' or not double_quote_appear):
                    ret = ret + ' '
                ret = ret + tok
            return ret.strip()

        pred_agg, pred_sel, pred_cond = pred_entry
        agg_score, sel_score, cond_score = score

        ret_queries = []
        if pred_agg:
            B = len(agg_score)
        elif pred_sel:
            B = len(sel_score)
        elif pred_cond:
            B = len(cond_score[0]) if reinforce else len(cond_score)
        for b in range(B):
            cur_query = {}
            if pred_agg:
                cur_query['agg'] = np.argmax(agg_score[b].data.cpu().numpy())
            if pred_sel:
                cur_query['sel'] = np.argmax(sel_score[b].data.cpu().numpy())
            if pred_cond:
                cur_query['conds'] = []
                all_toks = self.SQL_TOK + \
                           [x for toks in col[b] for x in
                            toks+[',']] + [''] + q[b] + ['']
                cond_toks = []
                if reinforce:
                    for choices in cond_score[1]:
                        if choices[b].data.cpu().numpy()[0] < len(all_toks):
                            cond_val = all_toks[choices[b].data.cpu().numpy()[0]]
                        else:
                            cond_val = '<UNK>'
                        if cond_val == '<END>':
                            break
                        cond_toks.append(cond_val)
                else:
                    for where_score in cond_score[b].data.cpu().numpy():
                        cond_tok = np.argmax(where_score)
                        cond_val = all_toks[cond_tok]
                        if cond_val == '<END>':
                            break
                        cond_toks.append(cond_val)

                if verbose:
                    print cond_toks
                if len(cond_toks) > 0:
                    cond_toks = cond_toks[1:]
                st = 0
                while st < len(cond_toks):
                    cur_cond = [None, None, None]
                    ed = len(cond_toks) if 'AND' not in cond_toks[st:] \
                         else cond_toks[st:].index('AND') + st
                    if 'EQL' in cond_toks[st:ed]:
                        op = cond_toks[st:ed].index('EQL') + st
                        cur_cond[1] = 0
                    elif 'GT' in cond_toks[st:ed]:
                        op = cond_toks[st:ed].index('GT') + st
                        cur_cond[1] = 1
                    elif 'LT' in cond_toks[st:ed]:
                        op = cond_toks[st:ed].index('LT') + st
                        cur_cond[1] = 2
                    else:
                        op = st
                        cur_cond[1] = 0
                    sel_col = cond_toks[st:op]
                    to_idx = [x.lower() for x in raw_col[b]]
                    pred_col = merge_tokens(sel_col, raw_q[b] + ' || ' + \
                                            ' || '.join(raw_col[b]))
                    if pred_col in to_idx:
                        cur_cond[0] = to_idx.index(pred_col)
                    else:
                        cur_cond[0] = 0
                    cur_cond[2] = merge_tokens(cond_toks[op+1:ed], raw_q[b])
                    cur_query['conds'].append(cur_cond)
                    st = ed + 1
            ret_queries.append(cur_query)

        return ret_queries
