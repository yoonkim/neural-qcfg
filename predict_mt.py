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

parser.add_argument('--data_file', default='data/MT/test-daxy.en')
parser.add_argument('--out_file', default='mt-pred-daxy.txt')
parser.add_argument('--model_path', default='mt.pt')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--num_samples', default=1000, type=int, help='samples')
parser.add_argument('--seed', default=3435, type=int, help='random seed')


def main(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  cuda.set_device(args.gpu)
  device = torch.device("cuda:"+str(args.gpu))
  data = open(args.data_file, "r")
  model_checkpoint = torch.load(args.model_path)
  encoder = model_checkpoint["encoder"]
  decoder = model_checkpoint["decoder"]
  parser = model_checkpoint["enc_parser"]
  x_tokenizer = model_checkpoint["x_tokenizer"]
  y_tokenizer = model_checkpoint["y_tokenizer"]
  model_args = model_checkpoint["args"]
  encoder.to(device)
  decoder.to(device)
  parser.to(device)
  out = open(args.out_file, "w")
  eval(data, encoder, decoder, parser, device, x_tokenizer, y_tokenizer, model_args, out)
  out.close()

def eval(data, encoder, decoder, parser, device, x_tokenizer, y_tokenizer, model_args, out):
  num_sents = 0
  num_words_pred = 0
  total_nll_pred = 0.
  for d in data:
    x = d.strip().split()
    x_len = len(x)
    x_tensor, _, _ = x_tokenizer.convert_batch([x])
    x_tensor = x_tensor.to(device)
    x_lengths = torch.Tensor([len(x)]).long().to(device)
    _, x_spans, _, x_actions, _ = parser(x_tensor, x_lengths)
    with torch.no_grad():
      node_features, node_spans = encoder(x_tensor, x_lengths,
                                          spans = x_spans)
      y_preds = decoder.decode(node_features, node_spans, y_tokenizer, 
                              num_samples = args.num_samples)
      best_pred = [""]
      best_nll = 1e5
      best_length = 0
      best_ppl = 1e5
      for y_pred in y_preds[0]:
        if len(y_pred) < 2:
          continue
        y_pred = [y_pred]
        y_pred_tensor, _, _  = y_tokenizer.convert_batch(y_pred)
        y_pred_tensor = y_pred_tensor.to(device)
        y_pred_lengths = torch.Tensor([len(y_pred[0])]).long().to(device)
        with torch.no_grad():
          if len(y_pred[0]) > 10:
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
            total_nll_pred += best_nll
            best_length = y_pred_lengths.sum().item()        
            num_words_pred += best_length
            y_pred_tree, pred_all_spans, pred_all_spans_node = decoder(
              y_pred_tensor, y_pred_lengths, node_features, node_spans,
              x_str=y_pred, argmax=True)

    pred = " ".join(best_pred)
    src = " ".join(x)
    out.write(pred +  "\n")
    x_parse = get_tree(x_actions[0], x)
    print("SRC PARSE: %s" % x_parse)
    print("SRC: %s\nPRED: %s" % (" ".join(x), pred))

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
