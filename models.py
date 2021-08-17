import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_struct import SentCFG
from utils import *
import torch_struct
import math
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical as Cat
from PCFG import PCFG

class ResidualLayer(nn.Module):
  def __init__(self, dim = 100):
    super(ResidualLayer, self).__init__()
    self.lin1 = nn.Linear(dim, dim)
    self.lin2 = nn.Linear(dim, dim)
  def forward(self, x):
    return F.relu(self.lin2(F.relu(self.lin1(x)))) + x

class MultiResidualLayer(nn.Module):
  def __init__(self, in_dim=100, res_dim = 100, out_dim=None, num_layers=3):
    super(MultiResidualLayer, self).__init__()
    self.num_layers = num_layers
    if in_dim is not None:
      self.in_linear = nn.Linear(in_dim, res_dim)
    else:
      self.in_linear = None
    if out_dim is not None:
      self.out_linear = nn.Linear(res_dim, out_dim)
    else:
      self.out_linear = None
    self.res_blocks = nn.ModuleList([ResidualLayer(res_dim) for _ in range(num_layers)])

  def forward(self, x):
    if self.in_linear is not None:
      out = self.in_linear(x)
    else:
      out = x
    for i in range(self.num_layers):
      out = self.res_blocks[i](out)
    if self.out_linear is not None:
      out = self.out_linear(out)
    return out

class NeuralQCFG(nn.Module):
  def __init__(self, vocab = 100,
               dim = 256, 
               num_layers = 3,
               src_dim = 256,
               pt_states = 0,
               nt_states = 0,
               src_nt_states = 0,
               src_pt_states = 0,
               dropout = 0.0, 
               rule_constraint_type = 2, 
               use_copy = False,
               tokenizer=None,
               nt_span_range = [0, 1000],
               pt_span_range = [0, 1000]):
    super(NeuralQCFG, self).__init__()
    self.pcfg = PCFG()
    self.vocab = vocab    
    self.src_dim = src_dim
    self.src_nt_states = src_nt_states
    self.src_pt_states = src_pt_states
    self.dim = dim
    self.nt_states = nt_states
    self.pt_states = pt_states
    self.src_nt_emb = nn.Parameter(torch.randn(src_nt_states, dim))
    self.register_parameter('src_nt_emb', self.src_nt_emb)
    self.src_nt_node_mlp = MultiResidualLayer(in_dim=src_dim, res_dim = dim, 
                                              num_layers=num_layers)
    self.src_pt_emb = nn.Parameter(torch.randn(src_pt_states, dim))
    self.register_parameter('src_pt_emb', self.src_pt_emb)
    self.src_pt_node_mlp = MultiResidualLayer(in_dim=src_dim, res_dim = dim, 
                                              num_layers=num_layers)
    if self.nt_states > 0:
      self.tgt_nt_emb = nn.Parameter(torch.randn(nt_states, dim))
      self.register_parameter('tgt_nt_emb', self.tgt_nt_emb)
    if self.pt_states > 0:
      self.tgt_pt_emb = nn.Parameter(torch.randn(pt_states, dim))
      self.register_parameter('tgt_pt_emb', self.tgt_pt_emb)

    self.rule_mlp_parent = MultiResidualLayer(in_dim=dim, 
                                              res_dim = dim, 
                                              num_layers=num_layers, 
                                              out_dim = None)
    self.rule_mlp_left = MultiResidualLayer(in_dim=dim, 
                                            res_dim = dim, 
                                            num_layers=num_layers, 
                                            out_dim = None)
    self.rule_mlp_right = MultiResidualLayer(in_dim=dim, 
                                             res_dim = dim, 
                                             num_layers=num_layers, 
                                             out_dim = None)
    self.root_mlp_child = nn.Linear(dim, 1, bias=False)
    self.vocab_out = MultiResidualLayer(in_dim=dim, res_dim = dim, 
                                        num_layers=num_layers, out_dim = vocab)
    self.neg_huge = -1e5
    self.rule_constraint_type = rule_constraint_type
    self.use_copy = use_copy
    self.nt_span_range = nt_span_range
    self.pt_span_range = pt_span_range
    self.tokenizer = tokenizer
    if tokenizer is None:
      self.PAD = 0
      self.UNK = 1
      self.BOS = 2
      self.EOS = 3
    else:
      self.PAD = self.tokenizer.vocab2idx["<pad>"]
      self.UNK = self.tokenizer.vocab2idx["<unk>"]
      self.BOS = self.tokenizer.vocab2idx["<s>"]
      self.EOS = self.tokenizer.vocab2idx["</s>"]


  def decode(self, node_features, spans, tokenizer, num_samples = 10, multigpu=False):
    params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes  = self.get_params(
      node_features, spans)
    terms, rules, roots = params[0], params[1], params[2]
    preds = self.pcfg.sampled_decoding(terms, rules, roots, 
                                       self.nt_spans, self.src_nt_states,
                                       self.pt_spans, self.src_pt_states,
                                       num_samples = num_samples,
                                       use_copy = self.use_copy, max_length = 100)
    pred_strings = []
    for pred in preds:      
      pred_strings.append([tokenizer.convert_to_string(pred[i][0]).split() for i in \
                          range(len(preds[0]))])
    return pred_strings


  def get_nt_copy_spans(self, x, span, x_str):
    bsz, N = x.size()
    copy_span = [None for _ in range(N)]    
    max_span = max([len(s) for s in span])
    for w in range(1, N):
      c = torch.zeros(bsz, 1, N-w, self.src_nt_states*max_span+self.nt_states).to(x.device)
      mask = torch.zeros_like(c)
      c2 = c[:, :, :, :self.src_nt_states*max_span].view(
        bsz, 1, N-w, self.src_nt_states, max_span)      
      mask2 = mask[:, :, :, :self.src_nt_states*max_span].view(
        bsz, 1, N-w, self.src_nt_states, max_span)
      c2[:, :, :, -1].fill_(self.neg_huge*10)
      mask2[:, :, :, -1].fill_(1.0)
      for b in range(bsz):
        l = N
        for i in range(l-w):
          j = i + w
          for k, s in enumerate(span[b]):
            if s[-1] is not None:
              copy_str = " ".join(s[-1])
              if " ".join(x_str[b][i:j+1]) == copy_str:
                c2[b, :, i, -1, k] = 0
      copy_span[w] = (c, mask)
    return copy_span

  def get_params(self, node_features, spans, x=None, x_str=None):
    batch_size = len(spans)
    pt_node_features, nt_node_features = [], []
    pt_spans, nt_spans = [], []
    for span, node_feature in zip(spans, node_features):      
      pt_node_feature = []
      nt_node_feature = []
      pt_span = []
      nt_span = []
      for i, s in enumerate(span):
        s_len = s[1]-s[0] + 1
        if (s_len >= self.nt_span_range[0] and s_len <= self.nt_span_range[1]):
          nt_node_feature.append(node_feature[i])
          nt_span.append(s)            
        if s_len >= self.pt_span_range[0] and s_len <= self.pt_span_range[1]:
          pt_node_feature.append(node_feature[i])
          pt_span.append(s)        
      if len(nt_node_feature) == 0:
        nt_node_feature.append(node_feature[-1])
        nt_span.append(span[-1])
      pt_node_features.append(torch.stack(pt_node_feature))
      nt_node_features.append(torch.stack(nt_node_feature))
      pt_spans.append(pt_span)
      nt_spans.append(nt_span)
    nt_node_features = pad_sequence(nt_node_features, batch_first=True, padding_value=0.0)
    pt_node_features = pad_sequence(pt_node_features, batch_first=True, padding_value=0.0)     
    pt_num_nodes = pt_node_features.size(1)
    nt_num_nodes = nt_node_features.size(1)
    device = nt_node_features.device
    self.pt_spans = pt_spans
    self.nt_spans = nt_spans
    nt_emb = []
    src_nt_node_emb = self.src_nt_node_mlp(nt_node_features)
    src_nt_emb = self.src_nt_emb.unsqueeze(0).expand(batch_size, self.src_nt_states, self.dim)
    src_nt_emb = src_nt_emb.unsqueeze(2) + src_nt_node_emb.unsqueeze(1)
    src_nt_emb = src_nt_emb.view(batch_size, self.src_nt_states*nt_num_nodes, -1)
    nt_emb.append(src_nt_emb)
    if self.nt_states > 0:
      tgt_nt_emb = self.tgt_nt_emb.unsqueeze(0).expand(batch_size, self.nt_states, self.dim)
      nt_emb.append(tgt_nt_emb)
    nt_emb = torch.cat(nt_emb, 1)    
    
    pt_emb = []
    src_pt_node_emb = self.src_pt_node_mlp(pt_node_features)
    src_pt_emb = self.src_pt_emb.unsqueeze(0).expand(batch_size, self.src_pt_states, self.dim)
    src_pt_emb = src_pt_emb.unsqueeze(2) + src_pt_node_emb.unsqueeze(1)
    src_pt_emb = src_pt_emb.view(batch_size, self.src_pt_states*pt_num_nodes, -1)
    pt_emb.append(src_pt_emb)
    if self.pt_states > 0:
      tgt_pt_emb = self.tgt_pt_emb.unsqueeze(0).expand(batch_size, self.pt_states, self.dim)
      pt_emb.append(tgt_pt_emb)
    pt_emb = torch.cat(pt_emb, 1)
    
    nt = nt_emb.size(1)
    pt = pt_emb.size(1)
    all_emb = torch.cat([nt_emb, pt_emb], 1) 
    roots = self.root_mlp_child(nt_emb)
    roots = roots.view(batch_size, -1)      
    roots += self.neg_huge
    for s in range(self.src_nt_states):
      roots[:, s*nt_num_nodes + nt_num_nodes - 1] -= self.neg_huge

    roots = F.log_softmax(roots, 1)

    rule_emb_parent = self.rule_mlp_parent(nt_emb) # b x nt_all x dm        
    rule_emb_left = self.rule_mlp_left(all_emb)
    rule_emb_right = self.rule_mlp_right(all_emb)

    rule_emb_child = rule_emb_left[:, :, None, :] + rule_emb_right[:, None, :, :]
    rule_emb_child = rule_emb_child.view(batch_size, (nt+pt)**2, self.dim)
    rules = torch.matmul(rule_emb_parent, rule_emb_child.transpose(1,2))
    rules = rules.view(batch_size, nt, nt + pt, nt + pt)

    src_nt = nt - self.nt_states
    src_pt = pt - self.pt_states
    tgt_nt = self.nt_states
    tgt_pt = self.pt_states

    src_nt_idx = slice(0, src_nt)
    src_pt_idx = slice(src_nt + tgt_nt, src_nt + tgt_nt + src_pt)
    tgt_nt_idx = slice(src_nt, src_nt + tgt_nt)
    tgt_pt_idx = slice(src_nt + tgt_nt + src_pt, src_nt + tgt_nt + src_pt + tgt_pt)

    if self.rule_constraint_type > 0:
      if self.rule_constraint_type == 1:
        mask = self.get_rules_mask1(batch_size, nt_num_nodes, pt_num_nodes,
                                    nt_spans, pt_spans, device)
      elif self.rule_constraint_type == 2:
        mask = self.get_rules_mask2(batch_size, nt_num_nodes, pt_num_nodes,
                                    nt_spans, pt_spans, device)

      rules[:, src_nt_idx, src_nt_idx, src_nt_idx] += mask[:, :, :src_nt, :src_nt]
      rules[:, src_nt_idx, src_nt_idx, src_pt_idx] += mask[:, :, :src_nt, src_nt:]
      rules[:, src_nt_idx, src_pt_idx, src_nt_idx] += mask[:, :, src_nt:, :src_nt]
      rules[:, src_nt_idx, src_pt_idx, src_pt_idx] += mask[:, :, src_nt:, src_nt:]

    if self.nt_states > 0:
      rules[:, tgt_nt_idx, src_nt_idx, src_nt_idx] += self.neg_huge
      rules[:, tgt_nt_idx, src_nt_idx, src_pt_idx] += self.neg_huge
      rules[:, tgt_nt_idx, src_pt_idx, src_nt_idx] += self.neg_huge
      rules[:, tgt_nt_idx, src_pt_idx, src_pt_idx] += self.neg_huge
      rules[:, tgt_nt_idx, tgt_nt_idx, src_nt_idx] += self.neg_huge
      rules[:, tgt_nt_idx, tgt_nt_idx, src_pt_idx] += self.neg_huge
      rules[:, tgt_nt_idx, tgt_pt_idx, src_nt_idx] += self.neg_huge
      rules[:, tgt_nt_idx, tgt_pt_idx, src_pt_idx] += self.neg_huge
      rules[:, tgt_nt_idx, src_nt_idx, tgt_nt_idx] += self.neg_huge
      rules[:, tgt_nt_idx, src_nt_idx, tgt_pt_idx] += self.neg_huge
      rules[:, tgt_nt_idx, src_pt_idx, tgt_nt_idx] += self.neg_huge
      rules[:, tgt_nt_idx, src_pt_idx, tgt_pt_idx] += self.neg_huge

    rules = rules 
    rules = rules.view(batch_size, nt, (nt+pt)**2).log_softmax(2).view(
      batch_size, nt, nt+pt, nt+pt)
    
    terms = F.log_softmax(self.vocab_out(pt_emb), 2)

    if x is not None:
      n = x.size(1)
      terms = terms.unsqueeze(1).expand(batch_size, n, pt, terms.size(2))
      x_expand = x.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
      terms = torch.gather(terms, 3, x_expand).squeeze(3)
      if self.use_copy:
        copy_pt = torch.zeros(batch_size, n, pt).fill_(self.neg_huge*0.1).to(device)
        copy_pt_view = copy_pt[:, :, :src_pt].view(
          batch_size, n, self.src_pt_states, pt_num_nodes)        
        for b in range(batch_size):
          for c, s in enumerate(pt_spans[b]):
            if s[-1] == None:
              continue
            copy_str = " ".join(s[-1])
            for j in range(n):
              if x_str[b][j] == copy_str:
                copy_pt_view[:, j, -1, c] = 0.0
        copy_mask = torch.zeros_like(copy_pt)
        copy_mask_view = copy_mask[:, :, :src_pt].view(
          batch_size, n, self.src_pt_states, pt_num_nodes)        
        copy_mask_view[:, :, -1].fill_(1.0)
        terms = terms*(1-copy_mask) + copy_pt*copy_mask
        copy_nt = self.get_nt_copy_spans(x, nt_spans, x_str)
      else:
        copy_nt = None
      params = (terms, rules, roots, None, None, copy_nt)
    else:
      params = (terms, rules, roots)

    return params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes

  def get_rules_mask1(self, batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device):
    nt = nt_num_nodes*self.src_nt_states
    pt = pt_num_nodes*self.src_pt_states
    nt_node_mask = torch.ones(batch_size, nt_num_nodes, nt_num_nodes).to(device)
    pt_node_mask = torch.ones(batch_size, nt_num_nodes, pt_num_nodes).to(device)

    def is_parent(parent, child):
      if child[0] >= parent[0] and child[1] <= parent[1]:
        return True
      else:
        return False

    for b, (pt_span, nt_span) in enumerate(zip(pt_spans, nt_spans)):      
      for i, parent_span in enumerate(nt_span):
        for j, child_span in enumerate(nt_span):
          if not(is_parent(parent_span, child_span)):
            nt_node_mask[b, i, j].fill_(0.0)
        for j, child_span in enumerate(pt_span):
          if not(is_parent(parent_span, child_span)):
            pt_node_mask[b, i, j].fill_(0.0)    

    nt_node_mask = nt_node_mask[:, None, :, None, :].expand(
      batch_size, self.src_nt_states, nt_num_nodes, self.src_nt_states, nt_num_nodes).contiguous()
    pt_node_mask = pt_node_mask[:, None, :, None, :].expand(
      batch_size, self.src_nt_states, nt_num_nodes, self.src_pt_states, pt_num_nodes).contiguous()
    
    nt_node_mask = nt_node_mask.view(batch_size, nt, nt)
    pt_node_mask = pt_node_mask.view(batch_size, nt, pt)
    node_mask = torch.cat([nt_node_mask, pt_node_mask], 2)
    node_mask = node_mask.unsqueeze(3)*node_mask.unsqueeze(2)    
    node_mask = node_mask.view(batch_size, nt, (nt+pt)**2)
    node_mask = (1.0 - node_mask)*self.neg_huge
    return node_mask.view(batch_size, nt, nt+pt, nt+pt)


  def get_rules_mask2(self, batch_size, nt_num_nodes, pt_num_nodes, nt_spans, pt_spans, device):
    nt = nt_num_nodes*self.src_nt_states
    pt = pt_num_nodes*self.src_pt_states
    bsz = batch_size
    src_nt = self.src_nt_states
    src_pt = self.src_pt_states 
    node_nt = nt_num_nodes
    node_pt = pt_num_nodes
    node_mask = torch.zeros(bsz, src_nt*node_nt, src_nt*node_nt + src_pt*node_pt,
                           src_nt*node_nt + src_pt*node_pt).to(device)

    nt_idx = slice(0, src_nt*node_nt)
    pt_idx = slice(src_nt*node_nt, src_nt*node_nt + src_pt*node_pt)

    nt_ntnt = node_mask[:, nt_idx, nt_idx, nt_idx].view(bsz, src_nt, node_nt, 
                                                        src_nt, node_nt, src_nt, node_nt)
    nt_ntpt = node_mask[:, nt_idx, nt_idx, pt_idx].view(bsz, src_nt, node_nt,
                                                        src_nt, node_nt, src_pt, node_pt)
    nt_ptnt = node_mask[:, nt_idx, pt_idx, nt_idx].view(bsz, src_nt, node_nt,
                                                        src_pt, node_pt, src_nt, node_nt)
    nt_ptpt = node_mask[:, nt_idx, pt_idx, pt_idx].view(bsz, src_nt, node_nt,
                                                        src_pt, node_pt, src_pt, node_pt)
    def is_parent(parent, child):
      if child[0] >= parent[0] and child[1] <= parent[1]:
        return True
      else:
        return False

    def is_strict_parent(parent, child):
      return is_parent(parent, child) and parent != child

    def span_len(span):
      return span[1] - span[0] + 1

    def covers(parent, child1, child2):
      return (span_len(parent) == (span_len(child1) + span_len(child2))) and \
        ((parent[0] == child1[0] and parent[1] == child2[1]) or \
         (parent[0] == child2[0] and parent[1] == child1[1]))

    def overlaps(span1, span2):
      return is_parent(span1, span2) or is_parent(span2, span1)

    for b, (pt_span, nt_span) in enumerate(zip(pt_spans, nt_spans)):      
      min_nt_span = min([span_len(s) for s in nt_span])
      for i, parent in enumerate(nt_span):
        if span_len(parent) == min_nt_span:              
          nt_ntnt[b, :, i, :, i, :, i].fill_(1.0)                
          for j, child in enumerate(pt_span):
            if is_strict_parent(parent, child):
              nt_ntpt[b, :, i, :, i, :, j].fill_(1.0)                
              nt_ptnt[b, :, i, :, j, :, i].fill_(1.0)    
        if span_len(parent) == 1:
          for j, child in enumerate(pt_span):
            if parent == child:
              nt_ptnt[b, :, i, :, j, :, i].fill_(1.0)
              nt_ntpt[b, :, i, :, i, :, j].fill_(1.0)
              nt_ptpt[b, :, i, :, j, :, j].fill_(1.0)
        for j, child1 in enumerate(nt_span):
          for k, child2 in enumerate(nt_span):         
            if covers(parent, child1, child2):
              nt_ntnt[b, :, i, :, j, :, k].fill_(1.0)
              nt_ntnt[b, :, i, :, k, :, j].fill_(1.0)
          for k, child2 in enumerate(pt_span):
            if covers(parent, child1, child2):
              nt_ntpt[b, :, i, :, j, :, k].fill_(1.0)
              nt_ptnt[b, :, i, :, k, :, j].fill_(1.0)            
        for j, child1 in enumerate(pt_span):
          for k, child2 in enumerate(pt_span):
            if covers(parent, child1, child2):
              nt_ptpt[b, :, i, :, j, :, k].fill_(1.0)
              nt_ptpt[b, :, i, :, k, :, j].fill_(1.0)

    node_mask = (1.0 - node_mask)*self.neg_huge

    return node_mask.contiguous().view(batch_size, nt, nt+pt, nt+pt)


  def forward(self, x, lengths, node_features, spans, x_str = None, argmax=False, multigpu=False):
    
    params, pt_spans, pt_num_nodes, nt_spans, nt_num_nodes = self.get_params(
      node_features, spans, x, x_str=x_str)
    out = self.pcfg(params, lengths, argmax)
    src_nt_states = self.src_nt_states*nt_num_nodes
    src_pt_states = self.src_pt_states*pt_num_nodes
    terms = params[0]
    if not argmax:
      return out
    else:
      tree, all_spans_state = out
      all_spans_node = []
      for b, (all_span, pt_span, nt_span) in \
          enumerate(zip(all_spans_state, pt_spans, nt_spans)):
        all_span_node = []
        for s in all_span:
          if s[0] == s[1]:
            if s[2] < src_pt_states:
              all_span_node.append(pt_span[s[2] % pt_num_nodes])
            else:
              all_span_node.append([-1, -1, s[2] - src_pt_states])
          else:
            if s[2] < src_nt_states:
              all_span_node.append(nt_span[s[2] % nt_num_nodes])
            else:
              all_span_node.append([-1, -1, s[2] - src_nt_states])
        all_spans_node.append(all_span_node)
      return tree, all_spans_state, all_spans_node

class BinaryTreeLSTMLayer(nn.Module):
  def __init__(self, dim = 200):
    super(BinaryTreeLSTMLayer, self).__init__()
    self.dim = dim
    self.linear = nn.Linear(dim*2, dim*5)

  def forward(self, x1, x2, e=None):
    #x = (h, c). h, c = b x dim. hidden/cell states of children
    #e = b x e_dim. external information vector
    if not isinstance(x1, tuple):
      x1 = (x1, None)    
    h1, c1 = x1 
    if x2 is None: 
      x2 = (torch.zeros_like(h1), torch.zeros_like(h1))
    elif not isinstance(x2, tuple):
      x2 = (x2, None)    
    h2, c2 = x2
    if c1 is None:
      c1 = torch.zeros_like(h1)
    if c2 is None:
      c2 = torch.zeros_like(h2)
    concat = torch.cat([h1, h2], 1)
    all_sum = self.linear(concat)
    i, f1, f2, o, g = all_sum.split(self.dim, 1)
    c = torch.sigmoid(f1)*c1 + torch.sigmoid(f2)*c2 + torch.sigmoid(i)*torch.tanh(g)
    h = torch.sigmoid(o)*torch.tanh(c)
    return (h, c)
  

class BinaryTreeLSTM(nn.Module):
  def __init__(self, vocab = 10,
               dim = 16,
               max_position = 256, 
               layers = 1,
               dropout = 0.0,
               token_type_emb = None):
    super(BinaryTreeLSTM, self).__init__()
    self.dim = dim
    self.word_emb = nn.Embedding(vocab, dim)
    self.tree_rnn = BinaryTreeLSTMLayer(dim)
    self.SHIFT = 0
    self.REDUCE = 1
    if layers > 0:
      self.lstm = nn.LSTM(dim, dim, bidirectional=True, batch_first=True, num_layers = layers)
      self.proj = nn.Linear(dim*2, dim, bias=False)
    else:
      self.lstm = None
    self.token_type_emb = token_type_emb

  def get_actions(self, spans, l):
    spans_set = set([(s[0], s[1]) for s in spans if s[0] < s[1]])
    actions = [self.SHIFT, self.SHIFT]
    stack = [(0, 0), (1, 1)]
    ptr = 2
    num_reduce = 0
    while ptr < l:
      if len(stack) >= 2:
        cand_span = (stack[-2][0], stack[-1][1])
      else:
        cand_span = (-1, -1)
      if cand_span in spans_set:
        actions.append(self.REDUCE)
        stack.pop()
        stack.pop()
        stack.append(cand_span)
        num_reduce += 1
      else:
        actions.append(self.SHIFT)
        stack.append((ptr, ptr))
        ptr += 1
    while len(actions) < 2*l - 1:
      actions.append(self.REDUCE)
    return actions

  def forward(self, x, lengths, spans=None, token_type=None):
    batch, length = x.size()
    device = x.device
    word_emb = self.word_emb(x)
    if token_type is not None:
      word_emb += self.token_type_emb(token_type)
    if self.lstm is not None:
      h, _ = self.lstm(word_emb)
      word_emb = self.proj(h)    
    word_emb = word_emb[:, :, None, :]
    node_features = []
    all_spans = []
    for b in range(batch):
      len_b = lengths[b].item()
      spans_b = [(i, i, -1) for i in range(len_b)]
      node_features_b = [word_emb[b][i] for i in range(len_b)]
      stack = []
      if len_b == 1:
        actions = []
      else:
        actions = self.get_actions(spans[b], len_b)
      ptr = 0      
      for action in actions:
        if action == self.SHIFT:
          stack.append([(word_emb[b][ptr], None), (ptr, ptr, -1)])
          ptr += 1
        else:
          right = stack.pop()
          left = stack.pop()
          new = self.tree_rnn(left[0], right[0])
          new_span = (left[1][0], right[1][1], -1)
          spans_b.append(new_span)
          node_features_b.append(new[0])
          stack.append([new, new_span])
      node_features.append(torch.cat(node_features_b, 0))
      all_spans.append(spans_b)
    self.actions = actions
    return node_features, all_spans

class NeuralPCFG(nn.Module):
  def __init__(self, vocab = 100,
               dim = 256, 
               pt_states = 40,
               nt_states = 40,
               num_layers = 2, 
               vocab_out = None):
    super(NeuralPCFG, self).__init__()
    self.dim = dim
    self.vocab = vocab
    self.pt_emb = nn.Parameter(torch.randn(pt_states, dim))
    self.nt_emb = nn.Parameter(torch.randn(nt_states, dim))
    self.root_emb = nn.Parameter(torch.randn(1, dim))
    self.nt_states = nt_states
    self.pt_states = pt_states
    self.all_states = nt_states + pt_states
    self.register_parameter('pt_emb', self.pt_emb)
    self.register_parameter('nt_emb', self.nt_emb)
    self.register_parameter('root_emb', self.root_emb)
    self.rule_mlp = nn.Sequential(nn.Linear(dim, self.all_states**2))
    self.root_mlp = MultiResidualLayer(in_dim=dim, res_dim = dim,
                                       num_layers=num_layers, out_dim = nt_states)
    if vocab_out is None:
      self.vocab_out = MultiResidualLayer(in_dim=dim, res_dim = dim,
                                          num_layers=num_layers, out_dim = vocab)
    else:
      self.vocab_out = vocab_out
    self.neg_huge = -1e5
    
  def forward(self, x, lengths, argmax=False, multigpu=False):
    #x : batch x n
    device = x.device
    batch_size, n = x.size()
    root_emb = self.root_emb.expand(batch_size, self.dim)
    roots = self.root_mlp(root_emb)
    roots = F.log_softmax(roots, 1)
    nt_emb = self.nt_emb.unsqueeze(0).expand(batch_size, self.nt_states, self.dim)
    pt_emb = self.pt_emb.unsqueeze(0).expand(batch_size, self.pt_states, self.dim)
    nt = nt_emb.size(1)
    pt = pt_emb.size(1)
    rules = self.rule_mlp(nt_emb)    
    rules = F.log_softmax(rules, 2)
    rules = rules.view(batch_size, nt, nt+pt, nt+pt)
    terms = F.log_softmax(self.vocab_out(pt_emb), 2)
    terms = terms.unsqueeze(1).expand(batch_size, n, pt, self.vocab)
    x_expand = x.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
    terms = torch.gather(terms, 3, x_expand).squeeze(3)
    params = (terms, rules, roots)
    dist = SentCFG(params, lengths)
    log_Z = dist.partition
    sample = dist._struct(torch_struct.SampledSemiring).marginals(
      dist.log_potentials, lengths=dist.lengths)
    log_prob = dist._struct().score(dist.log_potentials, sample) - log_Z
    argmax = dist.argmax
    argmax_spans, argmax_trees = extract_parses(argmax[-1], lengths.tolist(), inc=1)
    sample_spans, sample_trees = extract_parses(sample[-1], lengths.tolist(), inc=1)
    sample_actions = [get_actions(tree) for tree in sample_trees]    
    return sample_spans, argmax_spans, log_prob, sample_actions, -log_Z

  def marginals(self, x, lengths, argmax=False, multigpu=False):
    #x : batch x n
    device = x.device
    batch_size, n = x.size()
    root_emb = self.root_emb.expand(batch_size, self.dim)
    roots = self.root_mlp(root_emb)
    roots = F.log_softmax(roots, 1)
    nt_emb = self.nt_emb.unsqueeze(0).expand(batch_size, self.nt_states, self.dim)
    pt_emb = self.pt_emb.unsqueeze(0).expand(batch_size, self.pt_states, self.dim)
    nt = nt_emb.size(1)
    pt = pt_emb.size(1)
    rules = self.rule_mlp(nt_emb)    
    rules = F.log_softmax(rules, 2)
    rules = rules.view(batch_size, nt, nt+pt, nt+pt)
    terms = F.log_softmax(self.vocab_out(pt_emb), 2)
    terms = terms.unsqueeze(1).expand(batch_size, n, pt, self.vocab)
    x_expand = x.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
    terms = torch.gather(terms, 3, x_expand).squeeze(3)
    params = (terms, rules, roots)
    dist = SentCFG(params, lengths)
    log_Z = dist.partition
    marginals = dist.marginals[-1]
    return -log_Z, marginals.sum(-1)

  def argmax(self, x, lengths, argmax=False, multigpu=False):
    #x : batch x n
    device = x.device
    batch_size, n = x.size()
    root_emb = self.root_emb.expand(batch_size, self.dim)
    roots = self.root_mlp(root_emb)
    roots = F.log_softmax(roots, 1)
    nt_emb = self.nt_emb.unsqueeze(0).expand(batch_size, self.nt_states, self.dim)
    pt_emb = self.pt_emb.unsqueeze(0).expand(batch_size, self.pt_states, self.dim)
    nt = nt_emb.size(1)
    pt = pt_emb.size(1)
    rules = self.rule_mlp(nt_emb)    
    rules = F.log_softmax(rules, 2)
    rules = rules.view(batch_size, nt, nt+pt, nt+pt)
    terms = F.log_softmax(self.vocab_out(pt_emb), 2)
    terms = terms.unsqueeze(1).expand(batch_size, n, pt, self.vocab)
    x_expand = x.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
    terms = torch.gather(terms, 3, x_expand).squeeze(3)
    params = (terms, rules, roots)
    dist = SentCFG(params, lengths)
    spans_onehot = dist.argmax[-1]
    tags = dist.argmax[0].max(-1)[1]
    argmax_spans, tree = extract_parses(spans_onehot, lengths.tolist(), inc=1)
    return tree

  def forward_nll_argmax(self, x, lengths, argmax=False, multigpu=False):
    #x : batch x n
    device = x.device
    batch_size, n = x.size()
    root_emb = self.root_emb.expand(batch_size, self.dim)
    roots = self.root_mlp(root_emb)
    roots = F.log_softmax(roots, 1)
    nt_emb = self.nt_emb.unsqueeze(0).expand(batch_size, self.nt_states, self.dim)
    pt_emb = self.pt_emb.unsqueeze(0).expand(batch_size, self.pt_states, self.dim)
    nt = nt_emb.size(1)
    pt = pt_emb.size(1)
    rules = self.rule_mlp(nt_emb)    
    rules = F.log_softmax(rules, 2)
    rules = rules.view(batch_size, nt, nt+pt, nt+pt)
    terms = F.log_softmax(self.vocab_out(pt_emb), 2)
    terms = terms.unsqueeze(1).expand(batch_size, n, pt, self.vocab)
    x_expand = x.unsqueeze(2).expand(batch_size, n, pt).unsqueeze(3)
    terms = torch.gather(terms, 3, x_expand).squeeze(3)
    params = (terms, rules, roots)
    dist = SentCFG(params, lengths)
    log_Z = dist.partition
    argmax = dist.argmax
    argmax_spans, argmax_trees = extract_parses(argmax[-1], lengths.tolist(), inc=1)    
    return -log_Z, argmax_spans, argmax_trees
