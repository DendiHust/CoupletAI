#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/17 21:02
# @Author : Dendi
# @contact: msexuan@163.com
# @File : Layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    '''编码层'''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        '''

        :param d_model:     模型输入维度
        :param d_inner:     前馈层隐层维度
        :param n_head:      多头
        :param d_k:         键向量维度
        :param d_v:         值向量维度
        :param dropout:
        '''
        super(EncoderLayer, self).__init__()

        self.sef_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask = None, slf_attn_mask= None):

        enc_output, enc_slf_attn = self.sef_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # if non_pad_mask != None:
        #     enc_output *= non_pad_mask
        enc_output *= non_pad_mask
        enc_output = self.pos_ffn(enc_output)
        # if non_pad_mask != None:
        #     enc_output *= non_pad_mask
        enc_output *= non_pad_mask
        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    '''解码层'''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner,dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        # dec_output shape [batch_size, len_q, n_head * d_v]
        # dec_slf_attn shape [n_head * batch_size, len_q, d_v]
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn


