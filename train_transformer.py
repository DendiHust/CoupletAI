#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/12 17:10
# @Author  : 云帆
# @Site    : 
# @File    : train_seq2seq.py
# @Software: PyCharm

import dataset_pro
import argparse
from transformer.Models import Transformer
import torch
import torch.optim as optim
import torch.nn as nn
import time
import math
from tqdm import tqdm
import numpy as np
import logger

# pad index
SRC_PAD_IDX = dataset_pro.SHANG_LIAN.vocab.stoi['<pad>']
TGT_SOS_IDX = dataset_pro.XIA_LIAN.vocab.stoi['<sos>']
TGT_EOS_IDX = dataset_pro.XIA_LIAN.vocab.stoi['<eos>']

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--epoches", default=20, type=int)
    args.add_argument("--lr", default=0.001, type=float)
    args.add_argument("--max_len", default=32, type=int)
    args.add_argument("--sl_vocab_size", default=len(dataset_pro.SHANG_LIAN.vocab.stoi), type=int)
    args.add_argument("--xl_vocab_size", default=len(dataset_pro.XIA_LIAN.vocab.stoi), type=int)
    args.add_argument("--embedding_dim", default=256, type=int)
    args.add_argument("--model_dim", default=256, type=int)
    args.add_argument("--fp_inner_dim", default=1024, type=int)
    args.add_argument("--dropout", default=0.1, type=float)
    args.add_argument("--n_layers", default=3, type=float)
    args.add_argument("--n_head", default=4, type=float)
    args.add_argument("--d_k", default=64, type=float)
    args.add_argument("--d_v", default=64, type=float)
    args.add_argument("--gradient_clip", default=5.0, type=float)
    args.add_argument("--no_cuda", default=True, action='store_true')
    return args.parse_args()

def get_pos_ids(len_list, max_length):
    result = []
    for len_item in len_list:
        tep_list = [item for item in range(len_item)]
        tep_list += [SRC_PAD_IDX] * (max_length - len_item)
        result.append(tep_list)
    return np.array(result)


def train(model: Transformer, optimizer, criterion, clip, device):
    model.train()
    epoches_loss = 0
    for index, batch in tqdm(enumerate(dataset_pro.train_iter)):
        shang_lian, shang_lian_length = batch.shang_lian
        shang_lian = shang_lian.permute(1, 0).to(device)
        # shang_lian_length = shang_lian_length.permute(1, 0).to(device)
        shang_lian_length = shang_lian_length.numpy()
        shang_lian_pos = torch.LongTensor(get_pos_ids(shang_lian_length, shang_lian.shape[1])).to(device)
        xia_lian, xia_lian_length = batch.xia_lian
        xia_lian = xia_lian.permute(1, 0).to(device)
        xia_lian_length = xia_lian_length.numpy()
        xia_lian_pos = torch.LongTensor(get_pos_ids(xia_lian_length, xia_lian.shape[1])).to(device)

        optimizer.zero_grad()

        outputs = model(shang_lian, shang_lian_pos, xia_lian, xia_lian_pos)
        outputs = outputs.contiguous().view(-1, outputs.shape[-1])
        xia_lian = xia_lian.contiguous().view(-1)
        loss = criterion(outputs, xia_lian)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # print(loss.item())
        optimizer.step()

        epoches_loss += loss.item()
    result_loss = epoches_loss / len(dataset_pro.train_iter)

    return result_loss


def evaluate(model: Transformer, criterion, device):
    model.eval()
    epoches_loss = 0
    print('evaluate')
    with torch.no_grad():
        for index, batch in enumerate(dataset_pro.valid_iter):
            shang_lian, shang_lian_length = batch.shang_lian
            shang_lian = shang_lian.permute(1, 0).to(device)
            # shang_lian_length = shang_lian_length.permute(1, 0).to(device)
            shang_lian_length = shang_lian_length.numpy()
            shang_lian_pos = torch.LongTensor(get_pos_ids(shang_lian_length, shang_lian.shape[1])).to(device)
            xia_lian, xia_lian_length = batch.xia_lian
            xia_lian = xia_lian.permute(1, 0).to(device)
            xia_lian_length = xia_lian_length.numpy()
            xia_lian_pos = torch.LongTensor(get_pos_ids(xia_lian_length, xia_lian.shape[1])).to(device)

            outputs = model(shang_lian, shang_lian_pos, xia_lian, xia_lian_pos)
            outputs = outputs.contiguous().view(-1, outputs.shape[-1])
            xia_lian = xia_lian.contiguous().view(-1)
            loss = criterion(outputs, xia_lian)
            epoches_loss += loss.item()
    return epoches_loss / len(dataset_pro.valid_iter)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    args = get_args()


    device = torch.device('cuda' if args.no_cuda == False else 'cpu')
    transformer_model = Transformer(args.sl_vocab_size, args.xl_vocab_size,embedding_dim=args.embedding_dim,
                                    d_model=args.model_dim, d_inner=args.fp_inner_dim,
                                    n_layers=args.n_layers, n_head=args.n_head,
                                    d_k=args.d_k, d_v=args.d_v, dropout=args.dropout).to(device)

    # 优化器
    optimizer = optim.Adam(transformer_model.parameters())

    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=SRC_PAD_IDX)

    # N_EPOCHS = 10
    # CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(args.epoches):

        start_time = time.time()

        train_loss = train(transformer_model, optimizer, criterion,args.gradient_clip, device)
        valid_loss = evaluate(transformer_model, criterion, device)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if train_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(transformer_model.state_dict(), './models/transformer/transformer-model_{}.pt'.format(epoch + 1))

        logger.info('Epoch: {:02} | Time: {}m {}s'.format(epoch + 1, epoch_mins, epoch_secs))
        logger.info('\tTrain Loss: {:.3f} | Train PPL: {:7.3f}'.format(train_loss, math.exp(train_loss)))
        logger.info('\t Val. Loss: {:.3f} |  Val. PPL: {:7.3f}'.format(valid_loss, math.exp(valid_loss)))
