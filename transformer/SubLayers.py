#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/16 15:09
# @Author : Dendi
# @contact: msexuan@163.com
# @File : SubLayers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        '''

        :param q:       shape [n_head * batch_size, len_q, d_k]
        :param k:       shape [n_head * batch_size, len_k, d_k]
        :param v:       shape [n_head * batch_size, len_v, d_v]
        :param mask:
        :return:
        '''
        # attn shape [n_head * batch_size, len_q, len_k]
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        # output shape [n_head * batch_size, len_q, d_v]
        output = torch.bmm(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    '''
    前馈神经网络
    '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class MultiHeadAttention(nn.Module):
    '''多头注意力模型'''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        '''

        :param n_head:      '头'数
        :param d_model:     输入维度
        :param d_k:         键向量维度
        :param d_v:         值向量维度
        :param dropout:
        '''
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        # 产生查询 向量q，键向量k，值向量v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        '''
        计算多头注意力
        :param q:       用于产生，查询向量   [batch_size, len_q, d_model]
        :param k:       用于产生，键向量
        :param v:       用于产生，值向量
        :param mask:
        :return:
        '''

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        # q, k, v shape [batch_size, len_q, n_head, d_k]
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # q shape [n*b, len_q, d_k]
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        # k shape [n*b, len_k, d_k]
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        # v shape [n*b, len_v, d_v]
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        # mask = mask.repeat(n_head, 1, 1)
        # output shape [n_head * batch_size, len_q, d_v]
        # attn shape [n_head * batch_size, len_q, d_v]
        output, attn = self.attention(q, k, v, mask=None)
        # output shape [n_head, batch_size, len_q, d_v]
        output = output.view(n_head, sz_b, len_q, d_v)
        # output shape [batch_size, len_q, n_head * d_v]
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn




