#Norm!/usr/bin/env python3
import numpy as np
import itertools
import random
import torch
import nltk

def covered(l, r, spans):
  for span in spans:
    if span[0] <= l and span[1] >= r:
      return True
  return False

def extract_dag(spans, length):
  nodes = list(range(len(spans)))
  adj = []
  spans.sort(key = lambda x: x[1]-x[0])
  for i, (node, span) in enumerate(zip(nodes, spans)):
    l, r, t = span
    if r-l > 0:
      covered_spans = []
      for k in reversed(range(i)):
        l2, r2, t2 = spans[k]
        if l2 >= l and r2 <= r and not covered(l2, r2, covered_spans):
          adj.append((node, nodes[k]))
          covered_spans.append([l2, r2])            
  return nodes, adj
            

def extract_parse(span, length, inc=0):
    tree = [(i, str(i)) for i in range(length)]
    tree = dict(tree)
    spans = []
    N = span.shape[0]
    cover = span.nonzero()
    for i in range(cover.shape[0]):
        w, r, A = cover[i].tolist()
        w = w + inc 
        r = r + w 
        l = r - w 
        spans.append((l, r, A))
        if l != r:
            span = '({} {})'.format(tree[l], tree[r])
            tree[r] = tree[l] = span

    return spans, tree[0]

def extract_parses(matrix, lengths, kbest=False, inc=0):
    batch = matrix.shape[1] if kbest else matrix.shape[0]
    spans = []
    trees = []
    for b in range(batch):
        if kbest:
            span, tree = extract_parses(
                matrix[:, b], [lengths[b]] * matrix.shape[0], kbest=False, inc=inc
            ) 
        else:
            span, tree = extract_parse(matrix[b], lengths[b], inc=inc)
        trees.append(tree)
        spans.append(span)
    return spans, trees 


def all_binary_trees(n):
  #get all binary trees of length n
  def is_tree(tree, n):
    # shift = 0, reduce = 1
    if sum(tree) != n-1:
      return False
    stack = 0    
    for a in tree:
      if a == 0:
        stack += 1
      else:
        if stack < 2:
          return False
        stack -= 1
      if stack < 0:
        return False
    return True
  valid_tree = []
  num_shift = 0
  num_reduce = 0
  num_actions = 2*n - 1
  trees = map(list, itertools.product([0,1], repeat = num_actions-3))
  start = [0, 0] #first two actions are always shift
  end = [1] # last action is always reduce
  for tree in trees: 
    tree = start + tree + end
    if is_tree(tree, n):
      valid_tree.append(tree[::])
  return valid_tree

def get_actions(tree, SHIFT = 0, REDUCE = 1, OPEN='(', CLOSE=')'):
  #input tree in bracket form: ((A B) (C D))
  #output action sequence: S S R S S R R
  actions = []
  tree = tree.strip()
  i = 0
  num_shift = 0
  num_reduce = 0
  left = 0
  right = 0
  while i < len(tree):
    if tree[i] != ' ' and tree[i] != OPEN and tree[i] != CLOSE: #terminal      
      if tree[i-1] == OPEN or tree[i-1] == ' ':
        actions.append(SHIFT)
        num_shift += 1
    elif tree[i] == CLOSE:
      actions.append(REDUCE)
      num_reduce += 1
      right += 1
    elif tree[i] == OPEN:
      left += 1
    i += 1
  # assert(num_shift == num_reduce + 1)
  return actions

def get_tree(actions, sent = None, SHIFT = 0, REDUCE = 1):
  #input action and sent (lists), e.g. S S R S S R R, A B C D
  #output tree ((A B) (C D))
  stack = []
  pointer = 0
  if sent is None:
    sent = list(map(str, range((len(actions)+1) // 2)))
#  assert(len(actions) == 2*len(sent) - 1)
  if len(sent) == 1:
      return "(" + sent[0] + ")"
  for action in actions:
    if action == SHIFT:
      word = sent[pointer]
      stack.append(word)
      pointer += 1
    elif action == REDUCE:
      right = stack.pop()
      left = stack.pop()
      stack.append('(' + left + ' ' + right + ')')
  assert(len(stack) == 1)
  return stack[-1]
      
def get_spans(actions, SHIFT = 0, REDUCE = 1):
  sent = list(range((len(actions)+1) // 2))
  spans = []
  pointer = 0
  stack = []
  for action in actions:
    if action == SHIFT:
      word = sent[pointer]
      stack.append(word)
      pointer += 1
    elif action == REDUCE:
      right = stack.pop()
      left = stack.pop()
      if isinstance(left, int):
        left = (left, None)
      if isinstance(right, int):
        right = (None, right)
      new_span = (left[0], right[1])
      spans.append(new_span)
      stack.append(new_span)
  return spans

def get_stats(span1, span2):
  tp = 0
  fp = 0
  fn = 0
  for span in span1:
    if span in span2:
      tp += 1
    else:
      fp += 1
  for span in span2:
    if span not in span1:
      fn += 1
  return tp, fp, fn

def update_stats(pred_spans, gold_spans, stats):
  tp, fp, fn = get_stats(pred_spans, gold_spans)
  stats[0] += tp
  stats[1] += fp
  stats[2] += fn

def get_f1(stat):
  prec = stat[0] / (stat[0] + stat[1]) if stat[0] + stat[1] > 0 else 0.
  recall = stat[0] / (stat[0] + stat[2]) if stat[0] + stat[2] > 0 else 0.
  f1 = 2*prec*recall / (prec + recall)*100 if prec+recall > 0 else 0.
  return f1


def span_str(start = None, end = None):
  assert(start is not None or end is not None)
  if start is None:
    return ' '  + str(end) + ')'
  elif end is None:
    return '(' + str(start) + ' '
  else:
    return ' (' + str(start) + ' ' + str(end) + ') '    

def get_tree_from_binary_matrix(matrix, length):    
  sent = list(map(str, range(length)))
  n = len(sent)
  tree = {}
  for i in range(n):
    tree[i] = sent[i]
  for k in np.arange(1, n):
    for s in np.arange(n):
      t = s + k
      if t > n-1:
        break
      if matrix[s][t].item() == 1:
        span = '(' + tree[s] + ' ' + tree[t] + ')'
        tree[s] = span
        tree[t] = span
  return tree[0]
    

def get_nonbinary_spans(actions, SHIFT = 0, REDUCE = 1):
  spans = []
  stack = []
  pointer = 0
  binary_actions = []
  nonbinary_actions = []
  num_shift = 0
  num_reduce = 0
  for action in actions:
    # print(action, stack)
    if action == "SHIFT":
      nonbinary_actions.append(SHIFT)
      stack.append((pointer, pointer))
      pointer += 1
      binary_actions.append(SHIFT)
      num_shift += 1
    elif action[:3] == 'NT(':
      stack.append('(')            
    elif action == "REDUCE":
      nonbinary_actions.append(REDUCE)
      right = stack.pop()
      left = right
      n = 1
      while stack[-1] is not '(':
        left = stack.pop()
        n += 1
      span = (left[0], right[1])
      if left[0] != right[1]:
        spans.append(span)
      stack.pop()
      stack.append(span)
      while n > 1:
        n -= 1
        binary_actions.append(REDUCE)        
        num_reduce += 1
    else:
      assert False  
  assert(len(stack) == 1)
  assert(num_shift == num_reduce + 1)
  return spans, binary_actions, nonbinary_actions

def get_nonbinary_tree(sent, tags, actions):
  pointer = 0
  tree = []
  for action in actions:
    if action[:2] == "NT":
      node_label = action[:-1].split("NT")[1]
      node_label = node_label.split("-")[0]
      tree.append(node_label)
    elif action == "REDUCE":
      tree.append(")")
    elif action == "SHIFT":
      leaf = "(" + tags[pointer] + " " + sent[pointer] + ")"
      pointer += 1
      tree.append(leaf)
    else:
      assert(False)
  assert(pointer == len(sent))
  return " ".join(tree).replace(" )", ")")

def build_tree(depth, sen):
  assert len(depth) == len(sen)

  if len(depth) == 1:
    parse_tree = sen[0]
  else:
    idx_max = np.argmax(depth)
    parse_tree = []
    if len(sen[:idx_max]) > 0:
      tree0 = build_tree(depth[:idx_max], sen[:idx_max])
      parse_tree.append(tree0)
    tree1 = sen[idx_max]
    if len(sen[idx_max + 1:]) > 0:
      tree2 = build_tree(depth[idx_max + 1:], sen[idx_max + 1:])
      tree1 = [tree1, tree2]
    if parse_tree == []:
      parse_tree = tree1
    else:
      parse_tree.append(tree1)
  return parse_tree

def get_brackets(tree, idx=0):
  brackets = set()
  if isinstance(tree, list) or isinstance(tree, nltk.Tree):
    for node in tree:
      node_brac, next_idx = get_brackets(node, idx)
      if next_idx - idx > 1:
        brackets.add((idx, next_idx))
        brackets.update(node_brac)
      idx = next_idx
    return brackets, idx
  else:
    return brackets, idx + 1

def get_nonbinary_spans_label(actions, SHIFT = 0, REDUCE = 1):
  spans = []
  stack = []
  pointer = 0
  binary_actions = []
  num_shift = 0
  num_reduce = 0
  for action in actions:
    # print(action, stack)
    if action == "SHIFT":
      stack.append((pointer, pointer))
      pointer += 1
      binary_actions.append(SHIFT)
      num_shift += 1
    elif action[:3] == 'NT(':
      label = "(" + action.split("(")[1][:-1]
      stack.append(label)
    elif action == "REDUCE":
      right = stack.pop()
      left = right
      n = 1
      while stack[-1][0] is not '(':
        left = stack.pop()
        n += 1
      span = (left[0], right[1], stack[-1][1:])
      if left[0] != right[1]:
        spans.append(span)
      stack.pop()
      stack.append(span)
      while n > 1:
        n -= 1
        binary_actions.append(REDUCE)        
        num_reduce += 1
    else:
      assert False  
  assert(len(stack) == 1)
  assert(num_shift == num_reduce + 1)
  return spans, binary_actions

def is_next_open_bracket(line, start_idx):
    for char in line[(start_idx + 1):]:
        if char == '(':
            return True
        elif char == ')':
            return False
    raise IndexError('Bracket possibly not balanced, open bracket not followed by closed bracket')    

def get_between_brackets(line, start_idx):
    output = []
    for char in line[(start_idx + 1):]:
        if char == ')':
            break
        assert not(char == '(')
        output.append(char)    
    return ''.join(output)

def get_tags_tokens_lowercase(line):
    output = []
    line_strip = line.rstrip()
    for i in range(len(line_strip)):
        if i == 0:
            assert line_strip[i] == '('    
        if line_strip[i] == '(' and not(is_next_open_bracket(line_strip, i)): # fulfilling this condition means this is a terminal symbol
            output.append(get_between_brackets(line_strip, i))
    #print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    for terminal in output:
        terminal_split = terminal.split()
        # print(terminal, terminal_split)
        assert len(terminal_split) == 2 # each terminal contains a POS tag and word        
        output_tags.append(terminal_split[0])
        output_tokens.append(terminal_split[1])
        output_lowercase.append(terminal_split[1].lower())
    return [output_tags, output_tokens, output_lowercase]    

def get_nonterminal(line, start_idx):
    assert line[start_idx] == '(' # make sure it's an open bracket
    output = []
    for char in line[(start_idx + 1):]:
        if char == ' ':
            break
        assert not(char == '(') and not(char == ')')
        output.append(char)
    return ''.join(output)


def get_nonbinary_actions(line):
    output_actions = []
    line_strip = line.rstrip()
    i = 0
    max_idx = (len(line_strip) - 1)
    while i <= max_idx:
        assert line_strip[i] == '(' or line_strip[i] == ')'
        if line_strip[i] == '(':
            if is_next_open_bracket(line_strip, i): # open non-terminal
                curr_NT = get_nonterminal(line_strip, i)
                output_actions.append('NT(' + curr_NT + ')')
                i += 1  
                while line_strip[i] != '(': # get the next open bracket, which may be a terminal or another non-terminal
                    i += 1
            else: # it's a terminal symbol
                output_actions.append('SHIFT')
                while line_strip[i] != ')':
                    i += 1
                i += 1
                while line_strip[i] != ')' and line_strip[i] != '(':
                    i += 1
        else:
             output_actions.append('REDUCE')
             if i == max_idx:
                 break
             i += 1
             while line_strip[i] != ')' and line_strip[i] != '(':
                 i += 1
    assert i == max_idx  
    return output_actions
