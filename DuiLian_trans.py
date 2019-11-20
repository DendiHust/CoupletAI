#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/13 15:48
# @Author  : 云帆
# @Site    : 
# @File    : DuiLian_seq.py
# @Software: PyCharm
from seq2seq import EncoderLayer, DecoderLayer, AttentionLayer, Seq2Seq
import json
import argparse
# import dataset_pro
import torch
import pandas as pd
import beam_search_trans as beam_search
from transformer.Models import Transformer
import numpy as np

with open('./stoi/sl_char_2_id.txt', mode='r', encoding='utf8') as f:
    sl_stoi = json.load(f)

sl_itos = {v: k for k, v in sl_stoi.items()}

with open('./stoi/xl_char_2_id.txt', mode='r', encoding='utf8') as f:
    xl_stoi = json.load(f)

xl_itos = {v: k for k, v in xl_stoi.items()}

PAD_IDX = sl_stoi['<pad>']
SOS_IDX = xl_stoi['<sos>']
print(SOS_IDX)
EOS_IDX = xl_stoi['<eos>']


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--epoches", default=100, type=int)
    args.add_argument("--lr", default=0.001, type=float)
    args.add_argument("--max_len", default=32, type=int)
    args.add_argument("--sl_vocab_size", default=len(sl_stoi), type=int)
    args.add_argument("--xl_vocab_size", default=len(xl_itos), type=int)
    args.add_argument("--embedding_dim", default=256, type=int)
    args.add_argument("--model_dim", default=256, type=int)
    args.add_argument("--fp_inner_dim", default=512, type=int)
    args.add_argument("--dropout", default=0.1, type=float)
    args.add_argument("--n_layers", default=2, type=float)
    
    args.add_argument("--n_head", default=8, type=float)
    args.add_argument("--d_k", default=64, type=float)
    args.add_argument("--d_v", default=64, type=float)
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


def get_pos_ids(len_list, max_length):
    result = []
    for len_item in len_list:
        tep_list = [item for item in range(len_item)]
        tep_list += [PAD_IDX] * (max_length - len_item)
        result.append(tep_list)
    return np.array(result)


def predict_xl(text, model: Transformer, device, is_beam_search=False):
    input_id = get_input_char_index(text)
    # input_length = torch.LongTensor([len(input_id)]).to(device)
    input_tensor = torch.LongTensor([input_id]).to(device)
    batch_size, src_len = input_tensor.shape
    trg = input_tensor.new_full((batch_size, 1), model.sos_idx)
    src_mask, trg_mask = model.make_masks(input_tensor, trg)

    if is_beam_search == False:
        # while True:
        encoder_output = model.encoder(input_tensor, src_mask)
        step = 0
        result = []
        while step < 200:
            # print(step)
            output = model.decoder(trg, encoder_output, trg_mask, src_mask)
            output = torch.argmax(output[:, -1], dim=1)
            result.append(output.item())
            if output.numpy()[0] == EOS_IDX:
                break
            output = output.unsqueeze(1)
            trg = torch.cat((trg, output), dim=1)
            src_mask, trg_mask = model.make_masks(input_tensor, trg)
            step += 1
        # outpu_tensor = torch.argmax(output.squeeze(1), 1)
        ouput_str = get_output_char(result)
        return ouput_str
    else:
        
        target = beam_search.beam_decode(input_tensor, model, beam_with=5)
        print(target)
        print(len(target[0][0]))
        ouput_str = get_output_char(target[0][0][1:])
        return ouput_str


if __name__ == '__main__':
    args = get_args()
    # pad index

    device = torch.device('cuda' if args.no_cuda == False else 'cpu')
    transformer_model = Transformer(args.sl_vocab_size, args.xl_vocab_size, hid_dim=args.embedding_dim,
                                    pf_dim=args.fp_inner_dim, n_layers=args.n_layers, n_heads=args.n_head,
									dropout=args.dropout, device=device, SOS_IDX=SOS_IDX, PAD_IDX=PAD_IDX, EOS_IDX=EOS_IDX).to(
        device)
    transformer_model.load_state_dict(torch.load('./models-bak/transformer/1120/transformer-model_47.pt', map_location='cpu'))
    transformer_model.load_state_dict(torch.load('./models-bak/transformer/1120/transformer-model_47.pt', map_location='cpu'))
    transformer_model.eval()
    # text = '履霜坚冰至'
    # print(predict_xl(text, transformer_model, device, is_beam_search=False))
    df = pd.read_excel('./couplet/result-test.xlsx')
    df['transformer'] = df['上联'].apply(lambda x: predict_xl(x, transformer_model, device, is_beam_search=False))
    df['transformer_beam'] = df['上联'].apply(lambda x: predict_xl(x, transformer_model, device, is_beam_search=True))
    df.to_excel('./couplet/result-test.xlsx',index=False)
