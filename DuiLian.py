#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/13 15:48
# @Author  : 云帆
# @Site    : 
# @File    : DuiLian.py
# @Software: PyCharm
from seq2seq import EncoderLayer, DecoderLayer, AttentionLayer, Seq2Seq
import json
import argparse
# import dataset_pro
import torch
import pandas as pd

with open('./stoi/sl_char_2_id.txt', mode='r', encoding='utf8') as f:
    sl_stoi = json.load(f)

sl_itos = {v: k for k, v in sl_stoi.items()}

with open('./stoi/xl_char_2_id.txt', mode='r', encoding='utf8') as f:
    xl_stoi = json.load(f)

xl_itos = {v: k for k, v in xl_stoi.items()}


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--epoches", default=20, type=int)
    args.add_argument("--lr", default=0.001, type=float)
    args.add_argument("--max_len", default=32, type=int)
    args.add_argument("--sl_vocab_size", default=len(sl_stoi), type=int)
    args.add_argument("--xl_vocab_size", default=len(xl_stoi), type=int)
    args.add_argument("--embedding_dim", default=150, type=int)
    args.add_argument("--enc_hidden_dim", default=256, type=int)
    args.add_argument("--dec_hidden_dim", default=256, type=int)
    args.add_argument("--dropout", default=0.1, type=float)
    args.add_argument("--teacher_forcing_ratio", default=0.3, type=float)
    args.add_argument("--gradient_clip", default=5.0, type=float)
    args.add_argument("--no_cuda", default=True, action='store_true')
    return args.parse_args()


def get_input_char_index(text):
    result = [sl_stoi['<sos>']]
    result += [sl_stoi.get(item, sl_stoi['<unk>']) for item in text]
    result += [sl_stoi['<eos>']]
    return result


def get_output_char(result_id_list):
    result = []
    for item in result_id_list:
        if xl_itos[item] != '<eos>':
            result.append(xl_itos[item])
    return ''.join(result)


def predict_xl(text, model, device):
    input_id = get_input_char_index(text)
    input_length = torch.LongTensor([len(input_id)]).to(device)
    input_tensor = torch.LongTensor(input_id).unsqueeze(1).to(device)
    output_logits, _ = model(input_tensor, input_length, None, 0)
    outpu_tensor = torch.argmax(output_logits.squeeze(1), 1)
    ouput_str = get_output_char(outpu_tensor.numpy()[1:])
    return ouput_str


if __name__ == '__main__':
    args = get_args()
    # pad index
    PAD_IDX = sl_stoi['<pad>']
    SOS_IDX = xl_stoi['<sos>']
    print(SOS_IDX)
    EOS_IDX = xl_stoi['<eos>']

    device = torch.device('cuda' if args.no_cuda == False else 'cpu')
    encoder_layer = EncoderLayer(args.sl_vocab_size, args.embedding_dim, args.enc_hidden_dim, args.dec_hidden_dim,
                                 args.dropout).to(device)
    atten_layer = AttentionLayer(args.enc_hidden_dim, args.dec_hidden_dim).to(device)
    decoder_layer = DecoderLayer(args.xl_vocab_size, args.embedding_dim, args.enc_hidden_dim, args.dec_hidden_dim,
                                 args.dropout, atten_layer).to(device)
    seq2seq_model = Seq2Seq(encoder_layer, decoder_layer, PAD_IDX, SOS_IDX, EOS_IDX, device).to(device)

    seq2seq_model.load_state_dict(torch.load('./seq2seq_30_model.pt', map_location='cpu'))
    seq2seq_model.eval()
    # text = '心不明点灯何用'
    # print(predict_xl(text, seq2seq_model, device))
    df = pd.read_excel('./couplet/result-test.xlsx')
    df['seq2seq_下联_1113-2'] = df['上联'].apply(lambda x: predict_xl(x, seq2seq_model, device))
    df.to_excel('./couplet/result-test.xlsx',index=False)