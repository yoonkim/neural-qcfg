import torch
from torch import nn
from torch.distributions import Categorical as Cat
from utils import *
from torch_struct import SentCFG

class PCFG(nn.Module):    

  def forward(self, params, lengths, argmax=False):
    terms, rules, roots = params[0], params[1], params[2]
    batch_size, n, pt = terms.size()
    batch_size, nt, _, _ = rules.size()
    # roots_nt = roots[:, :nt]
    # roots_pt = roots[:, nt:]
    num_len1 = (lengths == 1).sum().item()
    CFG = SentCFG      
    dist = CFG(params, lengths= lengths)
    self.dist = dist
    if not argmax:
      log_Z = dist.partition
      return -log_Z
    else:
      spans_onehot = dist.argmax[-1]
      tags = dist.argmax[0].max(-1)[1]
      argmax_spans, tree = extract_parses(spans_onehot, lengths.tolist(), inc=1)      
      all_spans = []
      for b in range(batch_size):
        all_spans.append([(i, i, int(tags[b][i].item())) for i in range(lengths[b].item())]
                         + argmax_spans[b])
      return tree, all_spans

  def sample(self, terms, rules, roots,  
             nt_spans, src_nt_states, pt_spans, src_pt_states,
             use_copy =True, num_samples = 1, 
             max_length = 100, max_samples = 100, UNK = 1):
    device = terms.device
    pt, v = terms.size()
    nt, _, _ = rules.size()
    rules = rules.view(nt, (nt+pt)**2)
    terms = terms 
    roots_prob = Cat(roots.exp())
    rules_prob = [Cat(rules[s].exp()) for s in range(nt)]
    terms_prob = [Cat(terms[s].exp()) for s in range(pt)]    
    samples = []
    scores = []
    src_nt = len(nt_spans)*src_nt_states 
    src_pt = len(pt_spans)*src_pt_states
    nt_num_nodes = len(nt_spans)
    pt_num_nodes = len(pt_spans)
    for _ in range(num_samples):
      num_samples = 0
      nonterminals = [-1]
      preterminals = []      
      score = 0
      while nonterminals:
        s = nonterminals.pop()
        if s == -1:
          sample = roots_prob.sample().item()
          score += roots[sample].item()
          nonterminals.append(sample)
        else:
          if s < nt:
            num_samples += 1
            sample = rules_prob[s].sample().item()
            score += rules[s][sample].item()
            left = sample // (nt+pt)
            right = sample % (nt+pt)
            if use_copy and left < src_nt:
              src_nt_state = left // nt_num_nodes              
              src_node = left % nt_num_nodes
              if src_nt_state == src_nt_states - 1:
                preterminals.append(nt_spans[src_node][-1])
              else:
                nonterminals.append(left)                
            else:
              nonterminals.append(left)                
            if use_copy and right < src_nt:
              src_nt_state = right // nt_num_nodes
              src_node = right % nt_num_nodes
              if src_nt_state == src_nt_states - 1:                
                preterminals.append(nt_spans[src_node][-1])
              else:
                nonterminals.append(right)                
            else:
              nonterminals.append(right)                
          else:
            preterminals.append(s - nt)
        if len(preterminals) > max_length or num_samples > max_samples:
          break
      preterminals = preterminals[::-1]
      terminals = []      
      for s in preterminals:        
        if type(s) == list:
          for w in s:
            terminals.append(w)
        elif type(s) == str:
          terminals.append(s)
        else:
          src_pt_state = s // pt_num_nodes
          src_node = s % pt_num_nodes            
          if src_pt_state == src_pt_states - 1 and use_copy:
            sample = pt_spans[src_node][-1]
            for w in sample:
              terminals.append(w)
          else:
            sample = terms_prob[s].sample().item()          
            score += terms[s][sample].item()
            if use_copy and sample == UNK:
              # force <unk> tokens to copy
              sample = pt_spans[src_node][-1]
              for w in sample:
                terminals.append(w)              
            else:
              terminals.append(sample)
      samples.append(terminals)      
      scores.append(score) 
    return samples, scores
    
  def sampled_decoding(self, terms, rules, roots,
                       nt_spans, src_nt_states, pt_spans, src_pt_states,            
                       use_copy = True,  num_samples=10,  max_length = 100):
    batch_size, pt, v = terms.size()
    _, nt, _, _ = rules.size()
    device = terms.device
    preds = []
    nll = []
    for b in range(batch_size):
      samples, scores = self.sample(terms[b], rules[b], roots[b], 
                                    nt_spans[b], src_nt_states, 
                                    pt_spans[b], src_pt_states,
                                    use_copy = use_copy,
                                    num_samples = num_samples,
                                    max_length = max_length)
      sample_scores = [(sample, score) for sample, score in zip(samples, scores)]
      sample_scores.sort(key = lambda x: x[1], reverse=True)
      preds.append(sample_scores)
    return preds

