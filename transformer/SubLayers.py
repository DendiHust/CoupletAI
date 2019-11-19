#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/18 20:57
# @Author : Dendi
# @contact: msexuan@163.com
# @File : SubLayers.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    '''前馈神经网络'''

    def __init__(self, d_in, d_hid, dropout=0.1):

        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''

        :param x:   shape [batch_size, seq_length, d_in]
        :return: output shape [batch_size, seq_length, d_in]
        '''
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super(SelfAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert self.hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, q, k, v, mask=None):
        '''

        :param q:   shape [batch_size, seq_length, hid_dim]
        :param k:   shape [batch_size, seq_length, hid_dim]
        :param v:   shape [batch_size, seq_length, hid_dim]
        :param mask:
        :return:
        '''
        batch_size = q.shape[0]

        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # Q,K,V shape [batch_size, n_heads, seq_length, hid_dim // n_heads]

        Q = Q.contiguous().view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.contiguous().view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.contiguous().view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # energy [batch_size, n_heads, seq_length, seq_length]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        # attention [batch_size, n_heads, seq_length, seq_length]
        attention = self.dropout(torch.softmax(energy, dim=-1))
        # x [batch_size, n_heads, seq_length, hid_dim // n_heads]
        x = torch.matmul(attention, V)

        x = x.contiguous().permute(0, 2, 1, 3)
        # x [batch_size, seq_length, hid_dim]
        x = x.contiguous().view(batch_size, -1, self.n_heads * (self.hid_dim // self.n_heads))

        x = self.fc(x)
        # [batch_size, seq_length, hid_dim]
        return x
