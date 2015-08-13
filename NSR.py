#!/usr/bin/env python
# 07/02: 11:45-1:30, 2:45-3:45

import os
import sys
import subprocess
import shlex
import re
import nltk
from nltk.tree import Tree
from nltk.parse.dependencygraph import DependencyGraph

stanford_parser_dir = ''
for directory in os.listdir(os.getcwd()):
  if directory.startswith('stanford-parser'):
    stanford_parser_dir = ''

CLASSPATH = os.path.join(os.getcwd(), stanford_parser_dir, '*')
MODEL     = 'PCFG'
XMX       = '1g'
FILE      = None

i = 0
while i < len(sys.argv):
  if   sys.argv[i] == '-c' or sys.argv[i] == '--classpath':
    CLASSPATH = sys.argv.pop(i+1)
    # /home/tdozat/Scratch/Metrics/stanford-parser-full-2015-04-20/*
  elif sys.argv[i] == '-m' or sys.argv[i] == '--model':
    MODEL = sys.argv.pop(i+1)
  elif sys.argv[i] == '-x' or sys.argv[i] == '--mx':
    XMX = sys.argv.pop(i+1)
  i+=1

if len(sys.argv) > 1:
  FILE_IS_TEMP = False
  FILE = sys.argv[-1]
else:
  FILE_IS_TEMP = True
  i = 0
  while os.path.isfile('temp%d.txt' % i):
    i += 1
  FILE = 'temp%d.txt' % i
  with open(FILE, 'w') as f:
    f.write(sys.stdin.read())

depRegex = re.compile(r'([a-z:]*).(.*)-([0-9]*), (.*)-([0-9]*).')
cmd = 'java -mx%s -cp "%s" edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat "penn,typedDependencies" edu/stanford/nlp/models/lexparser/english%s.ser.gz %s' % (XMX, CLASSPATH, MODEL, FILE)
output = subprocess.check_output(shlex.split(cmd))
cTreeStrs = []
dTreeStrs = []
inTrees = False
for line in output.split('\n'):
  dep = depRegex.match(line)
  if len(line) > 0 and line.strip()[0] == '(':
    if not inTrees:
      cTreeStrs.append([])
      inTrees = True
    cTreeStrs[-1].append(line)
  elif dep:
    if inTrees:
      dTreeStrs.append([])
      inTrees = False
    dTreeStrs[-1].append([dep.group(4), '', dep.group(3), dep.group(1).upper()])

cTreeStrs = ['\n'.join(cTreeStr) for cTreeStr in cTreeStrs]
cTrees    = [Tree.fromstring(cTreeStr) for cTreeStr in cTreeStrs]

for cTree, dTreeStr in zip(cTrees, dTreeStrs):
  leaves = list(cTree.subtrees(lambda t: t.height() == 2))
  #print leaves
  i,j = 0,0
  while i < len(leaves) and j < len(dTreeStr):
    if leaves[i][0] != dTreeStr[j][0]:
      if j > 0 and dTreeStr[j][0] == dTreeStr[j-1][0]:
        dTreeStr[j] = [dTreeStr[j][0], leaves[i-1][0], dTreeStr[j][2], dTreeStr[j][3].upper()]
        i -= 1
      #else:
      #  dTreeStr.insert(j, [leaves[i][0], '', '0', 'punct'])
    dTreeStr[j][1] = leaves[i].label()
    i += 1
    j += 1
#  for leaf in leaves[i:]:
#    dTreeStr.append([leaf[0], leaf.label(), '0', 'punct'])
dTreeStrs = ['\n'.join(['\t'.join(node) for node in dTreeStr]) for dTreeStr in dTreeStrs]
for dTreeStr in dTreeStrs:
  #print dTreeStr
  DependencyGraph(dTreeStr)

dTrees = [DependencyGraph(dTreeStr, cell_separator='\t') for dTreeStr in dTreeStrs]


#***********************************************************************
#12:15 - 1:15, 1:30 - 2:00, 2:30 - 6:00, 9:15 - 9:30
#5:15 - 7:15
#12:00 - 1:00
#1:40 - 3:10, 3:30 - 4:15, 4:45 - 6:00

class MetricalTree(Tree):

  #=====================================================================
  def __init__(self, node, children=None):

    self._stress = 0
    self._cat    = 0
    super(MetricalTree, self).__init__(node, children)
    self._cat = self._label
    self._label = None
    # If children is None, the tree is read from node, and
    # all parents will be set during parsing.
    if children is not None:
      if self._cat == 'NP':
        try:
          tempFlag = None
          for i, child in enumerate(self[::-1]):
            if tempFlag == True and child._cat.startswith('NN'):
              child._stress = 0 
              tempFlag = False
            elif tempFlag == True and not child._cat.startswith('NN'):
              child._stress = -1
              self[len(self)-i]._stress = 0
              tempFlag = False
            else:
              child._stress = -1
              if tempFlag is None and child._cat.startswith('NN'):
                tempFlag = True
          if len(self) == 1:
            child._stress = 0
        except:
          raise ValueError('Malformed NP')

      if self._cat != 'NP' or tempFlag == None:
        try:
          tempFlag = None
          for i, child in enumerate(self[::-1]):
            if isinstance(child, MetricalTree):
              if tempFlag is None and not child._cat in ('.', ':', '-RRB-', '-LRB-'):
                child._stress = 0
                tempFlag = False
              else:
                child._stress = -1
        except:
          raise ValueError('Malformed Tree')

  #=====================================================================
  def _setstress(self, child, index, dry_run=False):

    # If the child's type is incorrect, then complain.
    if not isinstance(child, MetricalTree):
      raise TypeError('Can not insert a non-MetrricalTree '+
                      'into a MetricalTree')

    # Set child's parent pointer & index.
    if not dry_run:
      child._stress =  0 - (index != len(self)-1)
      child._label  = '%s/%d' % (child._cat, child._stress)

  #=====================================================================
  def make_absolute(self, parent_stress = 0):

    self._stress = parent_stress + self._stress
    for child in self:
      if isinstance(child, Tree):
        child.make_absolute(self._stress)
        child._label  = '%s/%d' % (child._cat, child._stress)

for cTree in cTrees:
  mTree = MetricalTree.convert(cTree)
  mTree.make_absolute()
  print mTree
  #for s in mTree.subtrees(lambda t: t.height() == 2):
  #  print s
  print ''
  
