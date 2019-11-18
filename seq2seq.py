#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/11 17:48
# @Author  : 云帆
# @Site    : 
# @File    : seq2seq.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class EncoderLayer(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedd_dim,
                 enc_hidden_dim,
                 dec_hidden_dim,
                 dropout):
        super(EncoderLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedd_dim)
        self.rnn = nn.GRU(embedd_dim, enc_hidden_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_input, src_input_length):
        '''
        :param src_input: shape [seq_max_length, batch_size]
        :param src_input_length shape [src_input_length, batch_size]
        :return:
        '''
        # embedded shape [seq_max_length, batch_size, embedd_dim]
        embedded = self.embedding(src_input)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_input_length)

        # outputs shape [seq_max_length, batch_size, enc_hidden_dim * 2]
        # hidden shape [n_layers * num_directions, batch_size, enc_hidden_dim]

        packed_outputs, hidden = self.rnn(packed_embedded)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = torch.tanh(self.fc(hidden))
        # outputs shape [seq_max_length, batch_size, enc_hidden_dim * 2]
        # hidden shape [batch_size, dec_hidden_dim]
        return outputs, hidden


class AttentionLayer(nn.Module):

    def __init__(self,
                 enc_hidden_dim,
                 dec_hidden_dim):
        super(AttentionLayer, self).__init__()

        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Parameter(torch.rand(dec_hidden_dim))

    def forward(self, dec_hidden, encoder_outputs, mask):
        '''
        解码器当前的输出 与 编码器所有时刻的输出 进行 attention
        :param dec_hidden: shape [batch_size, dec_hidden_dim]，编码器最后一个状态 或者 解码器前一个状态
        :param encoder_outputs:  [seq_length, batch_size, enc_hidden_dim * 2]，编码器所有时刻的输出
        :return:
        '''
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # hidden shape [batch_size, src_len, dec_hidden_dim]
        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)
        # encoder_outputs: [batch_size, seq_length, enc_hidden_dim * 2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # energy shape [batch_size, seq_length, dec_hidden_dim]
        energy = self.attn(torch.cat((dec_hidden, encoder_outputs), dim=2))
        energy = torch.tanh(energy)
        # energy shape [batch_size, dec_hidden_dim, seq_length]
        energy = energy.permute(0, 2, 1)

        # self.v shape [dec_hidden_dim]
        # v shape [batch_size, 1, dec_hidden_dim]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        # attention shape [batch_size, seq_length]
        attention = torch.bmm(v, energy).squeeze(1)

        attention = attention.masked_fill(mask==0, -1e10)

        return F.softmax(attention, dim=1)


class DecoderLayer(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 enc_hidden_dim,
                 dec_hidden_dim,
                 dropout,
                 attention):
        super(DecoderLayer, self).__init__()

        self.vocab_size = vocab_size
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU((enc_hidden_dim * 2) + embedding_dim, dec_hidden_dim)
        self.out = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim + embedding_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, hidden, encoder_outputs, mask):
        '''

        :param dec_input: <sos> 或者是 解码器前一个生成的字符
        :param hidden: 编码器最后一个状态 或者 解码器前一个状态
        :param encoder_outputs: 编码器的输出
        :return:
        '''
        # input shape [batch_size]
        # hidden shape [batch_size, dec_hidden_dim]
        # encoder_outputs shape [src_length, batch_size, enc_hidden_dim * 2]
        # input shape [1, batch_size]
        dec_input = dec_input.unsqueeze(0)
        # embedd shape [1, batch_size, embedding_dim]
        embedd = self.embedding(dec_input)
        # a shape [batch_size, src_length]
        a = self.attention(hidden, encoder_outputs, mask)
        # a shape [batch_size, 1, src_length]
        a = a.unsqueeze(1)
        # encoder_outputs shape [batch_size, src_length, enc_hidden_dim * 2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # weighted shape [batch_size, 1, enc_hidden_dim * 2]
        weighted = torch.bmm(a, encoder_outputs)
        # weighted shape [1, batch_szie, enc_hidden_dim * 2]
        weighted = weighted.permute(1, 0, 2)

        # rnn_input shape [1, batch_size, enc_hidden_dim * 2 + embedding_dim]
        rnn_input = torch.cat((embedd, weighted), dim=2)

        # output shape [1, batch_size, dec_hidden_dim]
        # hidden shape [1, batch_size, dec_hidden_dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        assert (output == hidden).all()
        # embedd shape [batch_size, embedding_dim]
        embedd = embedd.squeeze(0)
        # output shape [batch_size, dec_hidden_dim]
        output = output.squeeze(0)
        # weighted shape [batch_size, enc_hidden_dim * 2]
        weighted = weighted.squeeze(0)
        # output shape [batch_size, vocab_size]
        output = self.out(torch.cat((output, weighted, embedd), dim=1))

        return output, hidden.squeeze(0), a.squeeze(1)


class Seq2Seq(nn.Module):

    def __init__(self, encoder: EncoderLayer, decoder: DecoderLayer, pad_index, sos_index, eos_index, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_index = pad_index
        self.sos_index = sos_index
        self.eos_index = eos_index

    def create_mask(self, src_input):
        '''
        :param src_input: shape [batch_size, seq_length]
        :return:
        '''
        mask = (src_input != self.pad_index).permute(1, 0)
        # mask shape [batch_size, seq_legnth]
        return mask


    def forward(self, src_input, src_input_length, trg_input, teacher_forcing_ratio=0.3):
        # src_input shape [src_input_length, batch_size]
        # trg_input shape [trg_input_length, batch_size]

        if trg_input is None:
            assert teacher_forcing_ratio == 0, "Must be zero during inference"
            inference = True
            trg_input = torch.zeros((100, src_input.shape[1])).long().fill_(self.sos_idx).to(src_input.device)
        else:
            inference = False

        batch_size = src_input.shape[1]
        max_length = src_input.shape[0]

        trg_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(max_length, batch_size, trg_vocab_size).to(self.device)

        attentions = torch.zeros(max_length, batch_size,src_input.shape[0]).to(self.device)

        encoder_outputs, hidden = self.encoder(src_input, src_input_length)
        # first input to decoder is <sos>
        input = trg_input[0, :]

        mask = self.create_mask(src_input)

        for t in range(1, max_length):
            output, hidden, attention = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[t] = output
            attentions[t] = attention

            tearch_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            input = trg_input[t] if tearch_force else top1

            if inference and input.item() == self.eos_index:
                return outputs[:t], attentions[:t]

        return outputs, attentions
