#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/18 21:40
# @Author : Dendi
# @contact: msexuan@163.com
# @File : Layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer2.SubLayers import PositionwiseFeedForward, SelfAttention


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
        
