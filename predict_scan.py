
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

parser.add_argument('--data_file', default='data/SCAN/tasks_test_addprim_jump.txt')
parser.add_argument('--model_path', default='model-scan-addjump.pt')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--num_samples', default=10, type=int, help='num samples for decoding')
parser.add_argument('--seed', default=3435, type=int, help='random seed')


def get_data(data_file):
  data = []
  for d in open(data_file, "r"):
    src, tgt = d.split("IN: ")[1].split(" OUT: ")
    src = src.strip().split()
    tgt = tgt.strip().split()
    if len(src) == 1 or len(tgt) == 1:
      src = src + src
      tgt = tgt + tgt
    data.append({"src": src, "tgt": tgt})
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
  parser = model_checkpoint["parser"]
  x_tokenizer = model_checkpoint["x_tokenizer"]
  y_tokenizer = model_checkpoint["y_tokenizer"]
  model_args = model_checkpoint["args"]
  encoder.to(device)
  decoder.to(device)
  parser.to(device)
  eval(data, encoder, decoder, parser, device, x_tokenizer, y_tokenizer, model_args)

def eval(data, encoder, decoder, parser, device, x_tokenizer, y_tokenizer, model_args):
  num_sents = 0
  num_words = 0.
  num_words_pred = 0
  total_nll = 0.
  total_nll_pred = 0.
  total_correct = 0.
  num_examples = 0.
  for d in data:
    x = [d["src"]]
    y = [d["tgt"]]
    gold = " ".join(y[0])
    src = " ".join(x[0])
    x_tensor, _, _ = x_tokenizer.convert_batch(x)
    y_tensor, _, _ = y_tokenizer.convert_batch(y)
    x_tensor, y_tensor = x_tensor.to(device), y_tensor.to(device)      
    x_lengths = torch.Tensor([len(d["src"])]).long().to(device)
    y_lengths = torch.Tensor([len(d["tgt"])]).long().to(device)
    _, x_spans, _, x_actions, _ = parser(x_tensor, x_lengths)
    with torch.no_grad():
      node_features, node_spans = encoder(x_tensor, x_lengths,
                                          spans = x_spans)
      num_sents += 1
      num_words += y_lengths.sum().item()
      nll = decoder(y_tensor, y_lengths, 
                    node_features, node_spans, argmax=False)
      total_nll += nll.sum().item()
      y_preds = decoder.decode(node_features, node_spans, y_tokenizer, 
                               num_samples = args.num_samples)
      best_pred = [""]
      best_nll = 1e5
      best_length = 0
      best_ppl = 1e5
      num_examples += 1
      for y_pred in y_preds[0]:
        if len(y_pred) < 2:
          continue
        y_pred = [y_pred]
        y_pred_tensor, _, _  = y_tokenizer.convert_batch(y_pred)
        y_pred_tensor = y_pred_tensor.to(device)
        y_pred_lengths = torch.Tensor([len(y_pred[0])]).long().to(device)
        with torch.no_grad():
          if len(y_pred[0]) > 60:
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
      if " ".join(best_pred) == gold:
        total_correct += 1
      print(total_correct / num_examples, np.exp(total_nll / num_words), np.exp(total_nll_pred/num_words_pred))              
    pred = " ".join(best_pred)
    x_parse = get_tree(x_actions[0], x[0])
    print("X: %s" % x_parse)
    print("SRC: %s\nPRED: %s\nGOLD: %s" % (" ".join(x[0]), pred, gold))
    print("")
  print("Accuracy: %.4f" % (total_correct / num_examples))
if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
