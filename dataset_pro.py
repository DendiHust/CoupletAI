#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/12 17:09
# @Author  : 云帆
# @Site    : 
# @File    : dataset_pro.py
# @Software: PyCharm

import torchtext
from torchtext.data import Field, Example, Dataset
from torchtext.data import BucketIterator
from tqdm import tqdm
import json

shanglian_dir = './couplet/train/in-bak.txt'
xialian_dir = './couplet/train/out-bak.txt'
stoi_dir = './stoi/char_2_id.txt'
max_length = 32
batch_size = 32


def get_content_list(file_path):
    with open(file_path, mode='r', encoding='utf8') as f:
        return list(f.readlines())


def tokenize_func(text):
    text = str(text)
    return text.split(' ')


SHANG_LIAN = Field(tokenize=tokenize_func, init_token='<sos>', eos_token='<eos>', fix_length=max_length)
XIA_LIAN = Field(tokenize=tokenize_func, init_token='<sos>', eos_token='<eos>', fix_length=max_length)


def get_dataset():
    examples = []
    fields = [('shang_lian', SHANG_LIAN), ('xia_lian', XIA_LIAN)]
    shang_lian_list = get_content_list(shanglian_dir)
    xia_lian_list = get_content_list(xialian_dir)

    for shang_lian, xia_lian in tqdm(zip(shang_lian_list, xia_lian_list)):
        examples.append(Example.fromlist([shang_lian, xia_lian], fields))

    return Dataset(examples, fields)


train_data, valid_data = get_dataset().split(0.9)

SHANG_LIAN.build_vocab(train_data, min_freq=5)

with open(stoi_dir, mode='w', encoding='utf8') as f:
    json.dump(SHANG_LIAN.vocab.stoi, f)

train_iter, valid_iter = BucketIterator.splits(
    (train_data, valid_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.shang_lian)
)

if __name__ == '__main__':
    print(len(get_content_list(shanglian_dir)))
    print(SHANG_LIAN.vocab.stoi['<sos>'])
    print(SHANG_LIAN.vocab.stoi['<eos>'])
    print(SHANG_LIAN.vocab.stoi['<pad>'])
    print(SHANG_LIAN.vocab.stoi['<unk>'])
