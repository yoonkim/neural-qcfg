#!/usr/bin/env python3
import sys
import os

import argparse
import json
import random
import shutil
import copy
import pickle
import torch
from torch import cuda
import numpy as np
import time
import logging
from tokenizer import Tokenizer
from utils import *
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser()

parser.add_argument('--data_file', default='data/StylePTB/ATP/test.tsv')
parser.add_argument('--out_file', default='styleptb-pred-atp.txt')
parser.add_argument('--model_path', default='')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--num_samples', default=1000, type=int, help='samples')
parser.add_argument('--seed', default=3435, type=int, help='random seed')

def get_data(data_file):
  data = []
  for d in open(data_file):
    src, tgt = d.split("\t")
    if ";" in src:
      src, emph = src.strip().split(";")
      emph = emph.strip()
      src = src.strip().split()
      emph_mask = []
      for w in src:
        if w == emph:
          emph_mask.append(1)
        else:
          emph_mask.append(0)
      data.append({"src": src, "tgt": tgt.strip().split(), "emph_mask": emph_mask})
    else:
      data.append({"src": src.strip().split(), "tgt": tgt.strip().split()})
  return data

def main(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  cuda.set_device(args.gpu)
  device = torch.device("cuda:"+str(args.gpu))
  data = get_data(args.data_file)
  model_checkpoint = torch.load(args.model_path)  
  encoder = model_checkpoint["encoder"]
  decoder = model_checkpoint["decoder"]
  enc_parser = model_checkpoint["parser"]
  tokenizer = model_checkpoint["tokenizer"]
  model_args = model_checkpoint["args"]
  encoder.to(device)
  decoder.to(device)
  enc_parser.to(device)
  out = open(args.out_file, "w")
  eval(data, encoder, decoder, enc_parser, device, tokenizer, model_args, out)
  out.close()

def eval(data, encoder, decoder, enc_parser, device, tokenizer, model_args, out):
  num_sents = 0
  num_words_pred = 0
  total_nll_pred = 0.
  for d in data:
    x = [d["src"]]
    y = [d["tgt"]]
    x_tensor, _, _ = tokenizer.convert_batch(x)
    y_tensor, _, _ = tokenizer.convert_batch(y)
    x_tensor, y_tensor = x_tensor.to(device), y_tensor.to(device)
    emph_mask = torch.LongTensor(d["emph_mask"]).to(device) if "emph_mask" in d else None      
    x_lengths = torch.Tensor([len(d["src"])]).long().to(device)
    y_lengths = torch.Tensor([len(d["tgt"])]).long().to(device)
    _, x_spans, _, x_actions, _ = enc_parser(x_tensor, x_lengths)
    with torch.no_grad():
      node_features, node_spans = encoder(x_tensor, x_lengths, spans = x_spans,
                                          token_type = emph_mask)
      new_spans = []
      for span, x_str in zip(node_spans, x):
        new_span = []
        for s in span:
          new_span.append([s[0], s[1], x_str[s[0]:s[1]+1]])
        new_spans.append(new_span)
      node_spans = new_spans
      y_preds = decoder.decode(node_features, node_spans, tokenizer, 
                              num_samples = args.num_samples)
      best_pred = None
      best_nll = 1e5
      best_length = None
      best_ppl = 1e5
      best_derivation = None
      for k, y_pred in enumerate(y_preds[0]):
        y_pred = [y_pred]
        y_pred_tensor, _, _  = tokenizer.convert_batch(y_pred)
        y_pred_tensor = y_pred_tensor.to(device)
        y_pred_lengths = torch.Tensor([len(y_pred[0])]).long().to(device)
        with torch.no_grad():
          if len(y_pred[0]) > 30 or len(y_pred[0]) < 2:
            continue
          pred_nll = decoder(y_pred_tensor, y_pred_lengths, 
                             node_features, node_spans, argmax=False,
                             x_str = y_pred)
          ppl = np.exp(pred_nll.item() / y_pred_lengths.sum().item())
          # if pred_nll.item() < best_nll:
          if ppl < best_ppl:
            best_ppl = ppl
            best_pred = y_pred[0]
            best_nll = pred_nll.item()
            best_length = y_pred_lengths.sum().item()
            y_pred_tree, pred_all_spans, pred_all_spans_node = decoder(
              y_pred_tensor, y_pred_lengths, node_features, node_spans,
              x_str=y_pred, argmax=True)
      num_words_pred += best_length
      total_nll_pred += best_nll
      print(np.exp(total_nll_pred/num_words_pred))

    pred = " ".join(best_pred)
    gold = " ".join(y[0])
    src = " ".join(x[0])
    out.write(pred +  "\n")
    x_parse = get_tree(x_actions[0], x[0])
    print("X: %s" % x_parse)
    print("SRC: %s\nPRED: %s\nGOLD: %s" % (" ".join(x[0]), pred, gold))

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
