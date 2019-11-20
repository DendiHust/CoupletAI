#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/14 13:33
# @Author  : 云帆
# @Site    : 
# @File    : beam_search.py
# @Software: PyCharm

import operator
from queue import PriorityQueue
import torch
import torch.nn as nn
import torch.nn.functional as F
from seq2seq import Seq2Seq
from transformer.Models import Transformer


class BeamSearchNode(object):
    def __init__(self, decoder_state, previous_node, word_index, log_prob, length):
        '''

        :param decoder_state:
        :param previous_node:
        :param word_index:
        :param log_prob:
        :param length:
        '''
        self.decoder_hidden = decoder_state
        self.previous_node = previous_node
        self.word_index = word_index
        self.log_prob = log_prob
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0

        return self.log_prob / float(self.length - 1 + 1e-6) + alpha * reward


def beam_decode(src_input, src_input_length, model: Seq2Seq, beam_with=3, topk=1):
    '''

    :param src_input: 输入的 char_id，shape：[seq_pad_length, batch_size]
    :param src_input_length: src_input的非pad长度 shape：[batch_size]
    :param model: Seq2Seq模型
    :param beam_with:   beam search 宽度
    :param topk:    生成topk个句子
    :return:
    '''
    batch_size = src_input.shape[1]
    decode_result = []
    # 获得编码器的输出
    # encoder_outputs shape [seq_pad_length, batch_size, enc_hidden_dim * 2]
    # encoder_hidden shape [batch_size, dec_hidden_dim]
    encoder_outputs, encoder_hidden = model.encoder(src_input, src_input_length)
    mask = model.create_mask(src_input)
    for batch_index in range(batch_size):
        # 当前句子的编码器输出
        encoder_output_current = encoder_outputs[:, batch_index, :].unsqueeze(1)
        # 解码器的第一个输入隐状态是编码器最后一个时刻的状态
        decoder_hidden = encoder_hidden[batch_index].unsqueeze(0)
        # 解码器的第一个输入是 <sos>
        decoder_input = torch.LongTensor([model.sos_index]).to(model.device)
        # 当前句子的mask，用于attention计算
        mask_current = mask[batch_index].unsqueeze(0)

        #
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))
        # 优先队列
        nodes_queue = PriorityQueue()
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)

        # 将node加入到优先队列
        nodes_queue.put((-node.eval(), node))
        q_size = 1

        # 开始 beam search
        while True:
            if q_size > 200:
                break
            # 获得 best_node
            score, n = nodes_queue.get()
            decoder_input = n.word_index
            decoder_hidden = n.decoder_hidden

            if n.word_index.item() == model.eos_index and n.previous_node != None:
                endnodes.append((score, n))
                if len(endnodes) > number_required:
                    break
                else:
                    continue

            # 解码
            decoder_output, decoder_hidden, _ = model.decoder(decoder_input, decoder_hidden, encoder_output_current, mask_current)
            decoder_output = F.softmax(decoder_output, dim=1)
            # 获得 beam_with个可能
            log_prob, indexs = torch.topk(decoder_output, beam_with)

            next_nodes = []

            for new_k in range(beam_with):
                decoded_t = indexs[0][new_k].view(-1)
                log_p = log_prob[0][new_k].item()
                node = BeamSearchNode(decoder_hidden, n, decoded_t, log_p+n.log_prob, n.length + 1)
                score = - node.eval()
                next_nodes.append((score, node))

            for i in range(len(next_nodes)):
                score, nn = next_nodes[i]
                nodes_queue.put((score, nn))

            q_size += len(next_nodes) - 1

        if len(endnodes) == 0:
            endnodes = [nodes_queue.get() for _ in range(topk)]

        utterances = []
        i = 0
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            if i >= topk:
                break
            unterance = []
            unterance.append(n.word_index.numpy()[0])

            # 回溯
            while n.previous_node != None:
                n = n.previous_node
                unterance.append(n.word_index.numpy()[0])

            unterance = unterance[::-1]
            utterances.append(unterance)
            i+=1
        decode_result.append(utterances)
        return decode_result




if __name__ == '__main__':
    pass
