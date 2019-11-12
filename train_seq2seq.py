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
import torch
import torch.optim as optim
import torch.nn as nn
import time
import math


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
    args.add_argument("--gradient_clip", default=5.0, type=float)
    args.add_argument("--no_cuda", default=False, action='store_true')
    return args.parse_args()


def train(model: Seq2Seq, optimizer, criterion, clip, teacher_radio):
    model.train()
    epoches_loss = 0
    for index, batch in enumerate(dataset_pro.train_iter):
        shang_lian = batch.shang_lian
        xia_lian = batch.xia_lian

        optimizer.zero_grad()

        outputs = model(shang_lian, xia_lian, teacher_radio)
        outputs = outputs[1:].view(-1, outputs.shape[-1])
        xia_lian = xia_lian[1:].view(-1)
        loss = criterion(outputs, xia_lian)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # print(loss.item())
        optimizer.step()

        epoches_loss += loss.item()
    result_loss = epoches_loss / len(dataset_pro.train_iter)

    return result_loss


def evaluate(model:Seq2Seq, criterion):
    model.eval()
    epoches_loss = 0
    with torch.no_grad():
        for index, batch in enumerate(dataset_pro.valid_iter):
            shang_lian = batch.shang_lian
            xia_lian = batch.xia_lian

            output = model(shang_lian, xia_lian, 0)

            output = output[1:].view(-1, output.shape[-1])
            xia_lian = xia_lian.view(-1)

            loss = criterion(output, xia_lian)
            epoches_loss += loss
    return epoches_loss / len(dataset_pro.valid_iter)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs




if __name__ == '__main__':
    args = get_args()
    encoder_layer = EncoderLayer(args.vocab_size, args.embedding_dim, args.enc_hidden_dim, args.dec_hidden_dim,
                                 args.dropout)
    atten_layer = AttentionLayer(args.enc_hidden_dim, args.dec_hidden_dim)
    decoder_layer = DecoderLayer(args.vocab_size, args.embedding_dim, args.enc_hidden_dim, args.dec_hidden_dim,
                                 args.dropout, atten_layer)
    seq2seq_model = Seq2Seq(encoder_layer, decoder_layer, atten_layer)

    # 优化器
    optimizer = optim.Adam(seq2seq_model.parameters())
    # pad index
    PAD_IDX = dataset_pro.XIA_LIAN.vocab.stoi['<pad>']
    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    N_EPOCHS = 10
    CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(seq2seq_model,  optimizer, criterion,args.teacher_forcing_ratio, args.gradient_clip)
        valid_loss = evaluate(seq2seq_model,  criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(seq2seq_model.state_dict(), 'tut3-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')



