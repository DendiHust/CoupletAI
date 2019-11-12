#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/12 17:10
# @Author  : 云帆
# @Site    : 
# @File    : train_seq2seq.py
# @Software: PyCharm

import dataset_pro
import argparse
from seq2seq import EncoderLayer, DecoderLayer, AttentionLayer, Seq2Seq


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--epoches", default=20, type=int)
    args.add_argument("--lr", default=0.001, type=float)
    args.add_argument("--max_len", default=32, type=int)
    args.add_argument("--vocab_size", default=len(dataset_pro.SHANG_LIAN.vocab.stoi), type=int)
    args.add_argument("--embedding_dim", default=150, type=int)
    args.add_argument("--enc_hidden_dim", default=256, type=int)
    args.add_argument("--dec_hidden_dim", default=256, type=int)
    args.add_argument("--dropout", default=0.1, type=float)
    args.add_argument("--teacher_forcing_ratio", default=0.3, type=float)
    args.add_argument("--no_cuda", default=False, action='store_true')
    return args.parse_args()

def train(model, epoches):
    model.train()
    for e in range(epoches):
        for index, batch in enumerate(dataset_pro.train_iter):
            shang_lian = batch.shang_lian
            print(shang_lian)
            return


if __name__ == '__main__':
    args = get_args()
    encoder_layer = EncoderLayer(args.vocab_size, args.embedding_dim, args.enc_hidden_dim, args.dec_hidden_dim,
                                 args.dropout)
    atten_layer = AttentionLayer(args.enc_hidden_dim, args.dec_hidden_dim)
    decoder_layer = DecoderLayer(args.vocab_size, args.embedding_dim, args.enc_hidden_dim, args.dec_hidden_dim,
                                 args.dropout, atten_layer)
    seq2seq_model = Seq2Seq(encoder_layer, decoder_layer, atten_layer)
    train(seq2seq_model, args.epoches)
