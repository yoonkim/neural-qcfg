#!/usr/bin496/env python3
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
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser()

# Data 
parser.add_argument('--train_file_src', default='data/MT/train.en')
parser.add_argument('--train_file_tgt', default='data/MT/train.fr')
parser.add_argument('--dev_file_src', default='data/MT/dev.en')
parser.add_argument('--dev_file_tgt', default='data/MT/dev.fr')
parser.add_argument('--save_path', default='model.pt', help='where to save the model')
parser.add_argument('--min_freq', default=2, type=int)
parser.add_argument('--sent_max_length_x', default=20, type=int)
parser.add_argument('--sent_max_length_y', default=20, type=int)
# Encoder 
parser.add_argument('--enc_dim', default=512, type=int)
parser.add_argument('--enc_layers', default=0, type=int)
parser.add_argument('--enc_dropout', default=0.0, type=float)
# Decoder  
parser.add_argument('--src_pt_states', default=14, type=int)
parser.add_argument('--src_nt_states', default=14, type=int)
parser.add_argument('--dec_dim', default=512, type=int)
parser.add_argument('--dec_dropout', default=0.0, type=float)
parser.add_argument('--dec_layers', default=3, type=int)
parser.add_argument('--rule_constraint_type', default=2, type=int)
parser.add_argument('--dec_nt_span_min', default=1, type=int)
parser.add_argument('--dec_nt_span_max', default=100, type=int)
parser.add_argument('--dec_pt_span_min', default=1, type=int)
parser.add_argument('--dec_pt_span_max', default=100, type=int)
parser.add_argument('--use_copy', default=0, type=int)
# Parser 
parser.add_argument('--parser_pt_states', default=20, type=int)
parser.add_argument('--parser_nt_states', default=20, type=int)
parser.add_argument('--parser_dim', default=256, type=int)
# Optimization 
parser.add_argument('--num_epochs', default=15, type=int, help='number of training epochs')
parser.add_argument('--lr', default=5e-4, type=float, help='starting learning rate')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='l2 weight decay')
parser.add_argument('--max_grad_norm', default=3, type=float, help='gradient clipping parameter')
parser.add_argument('--beta1', default=0.75, type=float, help='beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=17, type=int, help='random seed')
parser.add_argument('--print_every', type=int, default=1000, help='print stats after N examples')
parser.add_argument('--print_trees', type=int, default=1, help='print trees')
parser.add_argument('--eval_every', type=int, default=1000, help='eval on dev set after N examples')
parser.add_argument('--update_every', type=int, default=32, help='grad update after N examples')

def get_data(src_file, tgt_file):
  data = []
  for src, tgt in zip(open(src_file, "r"), open(tgt_file, "r")):
    src = src.strip().split()
    tgt = tgt.strip().split()
    data.append({"src": src, "tgt": tgt})
  return data

def main(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  cuda.set_device(args.gpu)
  device = torch.device("cuda:"+str(args.gpu))
  train_data = get_data(args.train_file_src, args.train_file_tgt)
  val_data = get_data(args.dev_file_src, args.dev_file_tgt)
  x_tokenizer = Tokenizer()
  x_tokenizer.train([d["src"] for d in train_data], min_freq = args.min_freq)
  y_tokenizer = Tokenizer()
  y_tokenizer.train([d["tgt"] for d in train_data], min_freq = args.min_freq)
  from models import BinaryTreeLSTM as Encoder
  from models import NeuralQCFG as Decoder    
  from models import NeuralPCFG as Parser
  encoder = Encoder(vocab = len(x_tokenizer.vocab2idx),
                    dim = args.enc_dim,
                    dropout = args.enc_dropout,
                    layers = args.enc_layers)
  decoder = Decoder(vocab = len(y_tokenizer.vocab2idx),
                    dim = args.dec_dim,
                    num_layers = args.dec_layers,
                    src_dim = args.enc_dim,
                    src_pt_states = args.src_pt_states,
                    src_nt_states = args.src_nt_states, 
                    dropout = args.dec_dropout,
                    rule_constraint_type = args.rule_constraint_type,
                    use_copy = args.use_copy == 1,
                    nt_span_range = [args.dec_nt_span_min, args.dec_nt_span_max],
                    pt_span_range = [args.dec_pt_span_min, args.dec_pt_span_max])
  enc_parser = Parser(vocab = len(x_tokenizer.vocab2idx),
                      dim = args.parser_dim,
                      nt_states = args.parser_nt_states,
                      pt_states = args.parser_pt_states)

  model = torch.nn.ModuleList([encoder, decoder, enc_parser])
  for name, param in model.named_parameters():    
    if param.dim() > 1:
      xavier_uniform_(param)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                               betas = (args.beta1, args.beta2), 
                               weight_decay = args.weight_decay)
  best_val_ppl = 1e5
  epoch = 0
  b = 0
  model.to(device)
  model.train()
  while epoch < args.num_epochs:
    start_time = time.time()
    epoch += 1  
    print('Starting epoch %d' % epoch)
    train_nll = 0.
    src_nll = 0.
    num_sents = 0.
    num_src_words = 0.
    num_words = 0.
    random.shuffle(train_data)
    for d in train_data:
      if len(d["src"]) > args.sent_max_length_x or len(d["tgt"]) > args.sent_max_length_y or \
         len(d["src"]) < 2 or len(d["tgt"]) < 2:
          continue
      b += 1
      x = [d["src"]]
      y = [d["tgt"]]
      x_tensor, _, _ = x_tokenizer.convert_batch(x)
      y_tensor, _, _ = y_tokenizer.convert_batch(y)
      x_tensor, y_tensor = x_tensor.to(device), y_tensor.to(device)      
      x_lengths = torch.Tensor([len(d["src"])]).long().to(device)
      y_lengths = torch.Tensor([len(d["tgt"])]).long().to(device)
      parse_sample, parse_argmax, parse_log_prob, parse_actions, parse_nll = enc_parser(
        x_tensor, x_lengths)
      node_features, node_spans = encoder(x_tensor, x_lengths, spans = parse_sample)
      nll = decoder(y_tensor, y_lengths, node_features, node_spans,
                    x_str = y, argmax=False)
      dec_loss = nll.mean()
      (dec_loss / args.update_every).backward()
      train_nll += nll.sum().item()        
      with torch.no_grad():
        node_features_argmax, node_spans_argmax = encoder(x_tensor, x_lengths,
                                                          spans = parse_argmax)
        nll_argmax = decoder(y_tensor, y_lengths, node_features_argmax, node_spans_argmax, 
                             x_str = y, argmax=False)
        neg_reward = (nll - nll_argmax).detach().item()
      obj = (neg_reward*parse_log_prob).mean() + parse_nll.mean()
      (obj / args.update_every).backward()
      src_nll += parse_nll.sum().item()
      if b % args.update_every == 0:
        if args.max_grad_norm > 0:
          torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)      
          torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)      
          torch.nn.utils.clip_grad_norm_(enc_parser.parameters(), args.max_grad_norm)      
        optimizer.step()
        optimizer.zero_grad()
      num_sents += len(y)
      num_words += y_lengths.sum().item()
      num_src_words += x_lengths.sum().item()
      if b % args.print_every == 0:
        enc_param_norm = sum([p.norm()**2 for p in encoder.parameters()]).item()**0.5
        dec_param_norm = sum([p.norm()**2 for p in decoder.parameters()]).item()**0.5
        parser_param_norm = sum([p.norm()**2 for p in enc_parser.parameters()]).item()**0.5
        log_str = 'Epoch: %d, Batch: %d/%d, |EncParam|: %.4f, |DecParam|: %.4f, ' + \
                  '|SrcParserParam|: %.4f, ' + \
                   'LR: %.4f, SrcPPL: %.4f, PPL: %.4f, ValPPL: %.4f, ' + \
                  'Throughput: %.2f examples/sec'
        print("-"*80)
        print(log_str %
              (epoch, b, len(train_data), 
               enc_param_norm, dec_param_norm, parser_param_norm, 
               args.lr, np.exp(src_nll / num_src_words), 
               np.exp(train_nll / num_words), best_val_ppl, 
               num_sents / (time.time() - start_time)))
        print("-"*80)
        if args.print_trees == 1:
          print("")
          with torch.no_grad():
            y_tree, all_spans, all_spans_node = decoder(
              y_tensor, y_lengths, node_features, node_spans,
              x_str = y, argmax=True)
          x_str = [x_tokenizer.idx2vocab[idx] for idx in x_tensor[0].tolist()]
          y_str = [y_tokenizer.idx2vocab[idx] for idx in y_tensor[0].tolist()]
          x_length = x_lengths[0].item()
          y_length = y_lengths[0].item()
          print("Source: %s\nTarget: %s" % (" ".join(x_str), " ".join(y_str)))
          print("")
          print("Source Tree: %s" % get_tree(parse_actions[0], x_str))
          action = get_actions(y_tree[0])              
          print("QCFG Tree: %s" % get_tree(action, y_str))
          print("")
          for span, span_node in zip(all_spans[0], all_spans_node[0]):
            if span_node[0] == -1:
              if span[0] == span[1]:
                x_span = "T" + str(span_node[2])
              else:
                x_span = "NT" + str(span_node[2])
            else:
              x_span = " ".join(x_str[span_node[0]:span_node[1]+1])
            y_span = " ".join(y_str[span[0]:span[1]+1])
            if span[0] == span[1]:
              denom = len(decoder.pt_spans[0])
            else:
              denom = len(decoder.nt_spans[0])
            print((y_span, x_span, "N" + str(span[2] // denom)))
        if b % args.eval_every == 0 and epoch > 1:        
          print('--------------------------------')
          print('Checking validation perf...')    
          val_ppl = eval(val_data, encoder, decoder, enc_parser, device, 
                         x_tokenizer, y_tokenizer)
          print('--------------------------------')
          if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            checkpoint = {
              'args': args.__dict__,
              'encoder': encoder.cpu(),
              'decoder': decoder.cpu(),
              'enc_parser': enc_parser.cpu(),
              'x_tokenizer': x_tokenizer,
              'y_tokenizer': y_tokenizer,
            }
            print('Saving checkpoint to %s' % args.save_path)
            torch.save(checkpoint, args.save_path)
            model.to(device)

def eval(data, encoder, decoder, parser, device, x_tokenizer, y_tokenizer):
  encoder.eval()
  decoder.eval()
  parser.eval()
  num_sents = 0
  num_words = 0
  total_nll = 0.
  x_examples = []
  x_spans = []
  y_examples = []
  x_lengths = []
  y_lengths = []
  y_lengths_all = []
  b = 0
  for d in data:
    if len(d["src"]) > args.sent_max_length_x or len(d["tgt"]) > args.sent_max_length_y or \
       len(d["src"]) < 2 or len(d["tgt"]) < 2:
        continue
    b += 1
    x = [d["src"]]
    y = [d["tgt"]]
    x_tensor, _, _ = x_tokenizer.convert_batch(x)
    y_tensor, _, _ = y_tokenizer.convert_batch(y)
    x_tensor, y_tensor = x_tensor.to(device), y_tensor.to(device)

    x_lengths = torch.Tensor([len(d["src"])]).long().to(device)
    y_lengths = torch.Tensor([len(d["tgt"])]).long().to(device)
    parse_sample, parse_argmax, parse_log_prob, parse_actions, parse_nll = parser(
      x_tensor, x_lengths)
    with torch.no_grad():
      node_features, node_spans = encoder(x_tensor, x_lengths, spans = parse_argmax)
      new_spans = []
      for span, x_str in zip(node_spans, x):
        new_span = []
        for s in span:
          new_span.append([s[0], s[1], x_str[s[0]:s[1]+1]])
        new_spans.append(new_span)
      node_spans = new_spans
      nll = decoder(y_tensor, y_lengths, node_features, node_spans,
                    x_str = y, argmax=False)
    total_nll += nll.sum().item()
    num_words += y_lengths.sum().item()
  ppl = np.exp(total_nll / num_words)
  print('PPL: %.4f' % ppl)
  encoder.train()
  decoder.train()
  parser.train()
  return ppl

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
