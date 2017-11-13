import json
import torch
from sqlnet.utils import *
from sqlnet.model.seq2sql import Seq2SQL
from sqlnet.model.sqlnet import SQLNet
import numpy as np
import datetime

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true', 
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--ca', action='store_true',
            help='Use conditional attention.')
    parser.add_argument('--dataset', type=int, default=0,
            help='0: original dataset, 1: re-split dataset')
    parser.add_argument('--rl', action='store_true',
            help='Use RL for Seq2SQL.')
    parser.add_argument('--baseline', action='store_true', 
            help='If set, then test Seq2SQL model; default is SQLNet model.')
    parser.add_argument('--train_emb', action='store_true',
            help='Use trained word embedding for SQLNet.')
    args = parser.parse_args()

    N_word=300
    B_word=42
    if args.toy:
        USE_SMALL=True
        GPU=True
        BATCH_SIZE=15
    else:
        USE_SMALL=False
        GPU=True
        BATCH_SIZE=64
    TEST_ENTRY=(True, True, True)  # (AGG, SEL, COND)

    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(
                    args.dataset, use_small=USE_SMALL)

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
        load_used=True, use_small=USE_SMALL) # load_used can speed up loading

    if args.baseline:
        model = Seq2SQL(word_emb, N_word=N_word, gpu=GPU, trainable_emb = True)
    else:
        model = SQLNet(word_emb, N_word=N_word, use_ca=args.ca, gpu=GPU,
                trainable_emb = True)

    if args.train_emb:
        agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_name(args)
        print "Loading from %s"%agg_m
        model.agg_pred.load_state_dict(torch.load(agg_m))
        print "Loading from %s"%sel_m
        model.sel_pred.load_state_dict(torch.load(sel_m))
        print "Loading from %s"%cond_m
        model.cond_pred.load_state_dict(torch.load(cond_m))
        print "Loading from %s"%agg_e
        model.agg_embed_layer.load_state_dict(torch.load(agg_e))
        print "Loading from %s"%sel_e
        model.sel_embed_layer.load_state_dict(torch.load(sel_e))
        print "Loading from %s"%cond_e
        model.cond_embed_layer.load_state_dict(torch.load(cond_e))
    else:
        agg_m, sel_m, cond_m = best_model_name(args)
        print "Loading from %s"%agg_m
        model.agg_pred.load_state_dict(torch.load(agg_m))
        print "Loading from %s"%sel_m
        model.sel_pred.load_state_dict(torch.load(sel_m))
        print "Loading from %s"%cond_m
        model.cond_pred.load_state_dict(torch.load(cond_m))

    print "Dev acc_qm: %s;\n  breakdown on (agg, sel, where): %s"%epoch_acc(
            model, BATCH_SIZE, val_sql_data, val_table_data, TEST_ENTRY)
    print "Dev execution acc: %s"%epoch_exec_acc(
            model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB)
    print "Test acc_qm: %s;\n  breakdown on (agg, sel, where): %s"%epoch_acc(
            model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY)
    print "Test execution acc: %s"%epoch_exec_acc(
            model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB)
