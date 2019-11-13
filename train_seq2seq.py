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
from tqdm import tqdm


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--epoches", default=20, type=int)
    args.add_argument("--lr", default=0.001, type=float)
    args.add_argument("--max_len", default=32, type=int)
    args.add_argument("--sl_vocab_size", default=len(dataset_pro.SHANG_LIAN.vocab.stoi), type=int)
    args.add_argument("--xl_vocab_size", default=len(dataset_pro.XIA_LIAN.vocab.stoi), type=int)
    args.add_argument("--embedding_dim", default=150, type=int)
    args.add_argument("--enc_hidden_dim", default=256, type=int)
    args.add_argument("--dec_hidden_dim", default=256, type=int)
    args.add_argument("--dropout", default=0.1, type=float)
    args.add_argument("--teacher_forcing_ratio", default=0.3, type=float)
    args.add_argument("--gradient_clip", default=5.0, type=float)
    args.add_argument("--no_cuda", default=True, action='store_true')
    return args.parse_args()


def train(model: Seq2Seq, optimizer, criterion, clip, teacher_radio, device):
    model.train()
    epoches_loss = 0
    for index, batch in tqdm(enumerate(dataset_pro.train_iter)):
        shang_lian, shang_lian_length = batch.shang_lian
        shang_lian = shang_lian.to(device)
        shang_lian_length = shang_lian_length.to(device)
        xia_lian = batch.xia_lian.to(device)

        optimizer.zero_grad()

        outputs, _ = model(shang_lian, shang_lian_length, xia_lian, teacher_radio)
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


def evaluate(model: Seq2Seq, criterion, device):
    model.eval()
    epoches_loss = 0
    with torch.no_grad():
        for index, batch in enumerate(dataset_pro.valid_iter):
            shang_lian, shang_lian_length = batch.shang_lian
            shang_lian = shang_lian.to(device)
            shang_lian_length = shang_lian_length.to(device)
            xia_lian = batch.xia_lian.to(device)

            output, _ = model(shang_lian,shang_lian_length, xia_lian, 0)
            output = output[1:].view(-1, output.shape[-1])
            xia_lian = xia_lian[1:].view(-1)

            loss = criterion(output, xia_lian)
            epoches_loss += loss.item()
    return epoches_loss / len(dataset_pro.valid_iter)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    args = get_args()
    # pad index
    PAD_IDX = dataset_pro.SHANG_LIAN.vocab.stoi['<pad>']
    SOS_IDX = dataset_pro.XIA_LIAN.vocab.stoi['<sos>']
    EOS_IDX = dataset_pro.XIA_LIAN.vocab.stoi['<eos>']

    device = torch.device('cuda' if args.no_cuda == False else 'cpu')
    encoder_layer = EncoderLayer(args.sl_vocab_size, args.embedding_dim, args.enc_hidden_dim, args.dec_hidden_dim,
                                 args.dropout).to(device)
    atten_layer = AttentionLayer(args.enc_hidden_dim, args.dec_hidden_dim).to(device)
    decoder_layer = DecoderLayer(args.xl_vocab_size, args.embedding_dim, args.enc_hidden_dim, args.dec_hidden_dim,
                                 args.dropout, atten_layer).to(device)
    seq2seq_model = Seq2Seq(encoder_layer, decoder_layer, PAD_IDX, SOS_IDX, EOS_IDX, device).to(device)

    # 优化器
    optimizer = optim.Adam(seq2seq_model.parameters())

    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # N_EPOCHS = 10
    # CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(args.epoches):

        start_time = time.time()

        train_loss = train(seq2seq_model, optimizer, criterion, args.teacher_forcing_ratio, args.gradient_clip, device)
        valid_loss = evaluate(seq2seq_model, criterion, device)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(seq2seq_model.state_dict(), 'tut3-model.pt')

        print('Epoch: {:02} | Time: {}m {}s'.format(epoch + 1, epoch_mins, epoch_secs))
        print('\tTrain Loss: {:.3f} | Train PPL: {:7.3f}'.format(train_loss, math.exp(train_loss)))
        print('\t Val. Loss: {:.3f} |  Val. PPL: {:7.3f}'.format(valid_loss, math.exp(valid_loss)))
