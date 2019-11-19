#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/18 21:40
# @Author : Dendi
# @contact: msexuan@163.com
# @File : Layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.SubLayers import PositionwiseFeedForward, SelfAttention


class EncoderLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super(EncoderLayer, self).__init__()
        self.slf_attn = SelfAttention(hid_dim, n_heads, dropout, device)
        self.pos_forward = PositionwiseFeedForward(hid_dim, pf_dim, dropout)
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_input, src_mask):
        '''

        :param src_input:   shape [batch_size, src_length, hid_dim]
        :param src_mask:    shape [batch_size, src_length]
        :return:
        '''
        attentions = self.slf_attn(src_input, src_input, src_input, src_mask)
        attentions = self.dropout(self.dropout(attentions))
        src_input = self.layer_norm(src_input + attentions)

        output = self.layer_norm(src_input + self.dropout(self.pos_forward(src_input)))

        return output

class DecoderLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super(DecoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.slf_attn = SelfAttention(hid_dim, n_heads, dropout, device)
        self.enc_attn = SelfAttention(hid_dim, n_heads, dropout, device)
        self.pos_forward = PositionwiseFeedForward(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask, src_mask):
        '''

        :param trg:         [batch_size, trg_seq_length, hid_dim]
        :param src:         [batch_size, src_seq_length, hid_dim]
        :param trg_mask:    [batch_size, trg_seq_length]
        :param src_mask:    [batch_size, src_seq_length]
        :return:            [batch_size, ?, hid_dim]
        '''
        trg_slf_attn = self.slf_attn(trg, trg, trg, trg_mask)
        trg_slf_attn = self.dropout(trg_slf_attn)
        trg = self.layer_norm(trg + trg_slf_attn)

        trg_enc_attn = self.enc_attn(trg, src, src, src_mask)
        trg_enc_attn = self.dropout(trg_enc_attn)
        trg = self.layer_norm(trg + trg_enc_attn)

        pf = self.pos_forward(trg)
        pf = self.dropout(pf)

        output = self.layer_norm(trg + pf)
        return output



