#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/19 10:50
# @Author  : 云帆
# @Site    : 
# @File    : Models.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Layers import EncoderLayer, DecoderLayer


class Encoder(nn.Module):
    def __init__(self, vocab_size, hid_dim, n_layers, n_heads,
                 pf_dim, dropout, device):
        super(Encoder, self).__init__()
        # self.vocab_size = vocab_size
        # self.hid_dim = hid_dim
        # self.n_layers = n_layers
        # self.n_heads = n_heads
        # self.pf_dim = pf_dim
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        self.token_embedding = nn.Embedding(vocab_size, hid_dim)
        self.pos_embedding = nn.Embedding(200, hid_dim)

        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        pos = torch.arange(0, src.shape[1]).unsqueeze(0).repeat(src.shape[0], 1).to(self.device)

        src = self.dropout((self.token_embedding(src) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            src = layer(src, src_mask)

        return src


class Decoder(nn.Module):
    def __init__(self, vocab_size, hid_dim, n_layers, n_heads, pf_dim,
                 dropout, device):
        super(Decoder, self).__init__()

        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        self.token_embedding = nn.Embedding(vocab_size, hid_dim)
        self.pos_embedding = nn.Embedding(200, hid_dim)

        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)
        ])

        self.fc = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask, src_mask):
        pos = torch.arange(0, trg.shape[1]).unsqueeze(0).repeat(trg.shape[0], 1).to(self.device)

        trg = self.dropout((self.token_embedding(trg) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        return self.fc(trg)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size,
                 hid_dim, n_heads, n_layers, pf_dim,
                 dropout, device, SOS_IDX, PAD_IDX):
        self.device = device
        self.encoder = Encoder(src_vocab_size, hid_dim, n_layers, n_heads, pf_dim, dropout, device)
        self.decoder = Decoder(trg_vocab_size, hid_dim, n_layers, n_heads, pf_dim, dropout, device)
        self.sos_idx = SOS_IDX
        self.pad_idx = PAD_IDX


    def make_masks(self, src, trg):
        '''

        :param src:     [batch_size, src_seq_length]
        :param trg:     [batch_size, trg_seq_length]
        :return:
        '''
        # src_mask shape [batch_size, 1, 1, src_seq_length]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask shape [batch_size, 1, trg_seq_length, 1]
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)

        trg_len = trg.shape[1]
        # trg_sub_mask shape [trg_seq_length, trg_seq_length]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # trg_mask shape [batch_size, 1, trg_seq_length, trg_seq_length]
        trg_mask = trg_pad_mask & trg_sub_mask

        return src_mask, trg_mask

    def forward(self, src, trg):
        '''

        :param src:
        :param trg:
        :return:
        '''
        src_mask, trg_mask = self.make_masks(src, trg)
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, trg_mask, src_mask)

        return dec_output


