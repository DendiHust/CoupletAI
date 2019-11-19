#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/18 9:51
# @Author  : 云帆
# @Site    : 
# @File    : Models.py
# @Software: PyCharm
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer


MAX_SEQ_LENGTH = 100


def get_attn_key_pad_mask(seq_k, seq_q, PAD_INDEX=1):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD_INDEX)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None, device=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table).to(device)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


def get_non_pad_mask(seq, PAD_INDEX=1):
    assert seq.dim() == 2
    return seq.ne(PAD_INDEX).type(torch.float).unsqueeze(-1)


class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_layers, n_head,
                 d_k, d_v, d_model, d_inner, dropout=0.1):
        '''

        :param vocab_size:      词汇表大小
        :param embedding_dim:   词向量维度
        :param n_layers:        编码器层数
        :param n_head:          “头”数量
        :param d_k:             键向量维度
        :param d_v:             值向量维度
        :param d_model:         模型输入维度
        :param d_inner:         前馈网络隐藏层维度
        :param dropout:
        '''
        super(Encoder, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(MAX_SEQ_LENGTH + 1, embedding_dim, padding_idx=0),
            freeze=True
        )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, src_seq, src_pos, return_attns=False):
        enc_slf_attn_list = []

        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        enc_output = self.word_embedding(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Decoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_layers,
                 n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super(Decoder, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(MAX_SEQ_LENGTH + 1, embedding_dim, padding_idx=0),
            freeze=True
        )

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
        '''

        :param tgt_seq:         shape [batch_size, tgt_seq_length]
        :param tgt_pos:         shape [batch_size, tgt_seq_length]
        :param src_seq:         shape [batch_size, src_seq_length]
        :param enc_output:      shape [batch_size, ]
        :param return_attns:
        :return:
        '''
        dec_slf_attn_list, dec_enc_attn_list = [], []

        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        dec_output = self.word_embedding(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask
            )

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list

        return dec_output


class Transformer(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 embedding_dim=150, d_model=512, d_inner=1024,
                 n_layers=3, n_head=4, d_k=64, d_v=64, dropout=0.1, device=None):
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.encoder = Encoder(src_vocab_size, embedding_dim, n_layers, n_head
                               , d_k, d_v, d_model, d_inner, dropout=dropout)

        self.decoder = Decoder(tgt_vocab_size, embedding_dim, n_layers, n_head,
                               d_k, d_v, d_model, d_inner, dropout=dropout)

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
        # tgt_seq, tgt_pos = tgt_seq[:, : -1], tgt_pos[:, : -1]

        # enc_output, *_ = self.encoder(src_seq, src_pos)
        enc_output = self.encoder(src_seq, src_pos)
        # dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)

        seq_logit = self.fc(dec_output)

        return seq_logit.view(-1, seq_logit.size()[2])
