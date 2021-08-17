#!/usr/bin/env python3
import numpy as np
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence

class Tokenizer(object):
  def __init__(self, vocab2idx=None):
    self.vocab2idx = None
    self.idx2vocab = None
    self.vocab_freq = None
    self.PAD = "<pad>"
    self.UNK = "<unk>"
    self.BOS = "<s>"
    self.EOS = "</s>"

    if vocab2idx is None:
      self.vocab2idx = {self.PAD: 0, self.UNK: 1, self.BOS: 2, self.EOS: 3}
    else:
      self.vocab2idx = vocab2idx
        
  def train(self, data, use_char=False, min_freq=0):
    self.vocab_freq = {}
    for d in data:
      for w in d:
        if use_char:
          for c in w:
            if c not in self.vocab_freq:
              self.vocab_freq[c] = 0
            self.vocab_freq[c] += 1
        else:
          if w not in self.vocab_freq:
            self.vocab_freq[w] = 0
          self.vocab_freq[w] += 1
    for key, count in self.vocab_freq.items():
      if count >= min_freq:
        self.vocab2idx[key] = len(self.vocab2idx)
    self.idx2vocab = {idx:key for key,idx in self.vocab2idx.items()}
    print("Num sents: %d, Vocab Size Before prune: %d, After prune: %d" %
          (len(data), len(self.vocab_freq), len(self.vocab2idx)))

  
  def convert_to_string(self, x_idx):
    out = []
    for idx in x_idx:
      if type(idx) == str:
        out.append(idx)
        continue
      x = self.idx2vocab[idx]
      if x == self.EOS:
        break
      if x != self.BOS:
        out.append(x)
    return " ".join(out)
    
  def convert(self, x, vocab2idx, UNK=None, BOS=None, EOS=None, max_length=None):
    x_idx = [vocab2idx[w] if w in vocab2idx else vocab2idx[UNK] for w in x]
    if BOS:
      x_idx = [vocab2idx[BOS]] + x_idx
    if EOS:
      x_idx = x_idx + [vocab2idx[EOS]]
    if max_length and len(x_idx) > max_length:
      if EOS:
        x_idx[max_length-1] = vocab2idx[EOS]
      x_idx = x_idx[:max_length]
    return torch.Tensor(x_idx).long()

  def convert_batch(self, sents, use_char=False, char_max_length=30, space="", 
                    use_bos_eos=False, use_char_bos_eos=False):
    sents_idx = []
    if not use_char:
      batch_vocab2idx = self.vocab2idx
      batch_word_onehot = []
      for sent in sents:
        if use_bos_eos:
          sent = [self.BOS] + sent + [self.EOS]
        sents_idx.append(self.convert(sent, self.vocab2idx, self.UNK))        
    else:
      batch_vocab2idx = {} 
      batch_word_onehot = []      
      # batch_vocab2idx = {self.PAD : 0} 
      # batch_word_onehot = [self.convert(list(self.PAD), self.vocab2idx, 
      #                                   self.UNK, self.BOS, self.EOS, char_max_length)]      
      for sent in sents:
        if use_bos_eos:
          sent = ["<s>"] + sent + ["</s>"]
        new_sent = []
        for w in sent:
          w_str = space.join(w)
          if w_str not in batch_vocab2idx:
            batch_vocab2idx[w_str] = len(batch_vocab2idx)
            if use_char_bos_eos:
              w_idx = self.convert(w, self.vocab2idx, 
                                   self.UNK, self.BOS, self.EOS, char_max_length)
            else:
              w_idx = self.convert(w, self.vocab2idx, 
                                   self.UNK, None, None, char_max_length)
            batch_word_onehot.append(w_idx)
          new_sent.append(w_str)
        sents_idx.append(self.convert(new_sent, batch_vocab2idx, self.UNK))
      batch_word_onehot = pad_sequence(batch_word_onehot, batch_first=True,
                                       padding_value = self.vocab2idx[self.PAD])
    sents_tensor = pad_sequence(
      sents_idx, batch_first=True, 
      padding_value = self.vocab2idx[self.PAD] if self.PAD in self.vocab2idx else -1)    
    return sents_tensor, batch_vocab2idx, batch_word_onehot
