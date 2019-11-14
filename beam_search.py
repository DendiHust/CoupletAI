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


class BeamSearchNode(object):
    def __init__(self, decoder_state, previous_node, word_index, log_prob, length):
        '''

        :param decoder_state:
        :param previous_node:
        :param word_index:
        :param log_prob:
        :param length:
        '''
        self.decoder_state = decoder_state
        self.previous_node = previous_node
        self.word_index = word_index
        self.log_prob = log_prob
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0

        return self.log_prob / float(self.length - 1 + 1e-6) + alpha * reward


def beam_decode(decode_outputs, encoder_outputs=None, beam_with=3):
    '''

    :param decode_outputs: 解码器
    :param encoder_outputs:
    :param beam_with:
    :return:
    '''
    pass



if __name__ == '__main__':
    pass

