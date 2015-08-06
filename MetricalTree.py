#!/usr/bin/env python
# 07/29/15: 12:15-1:15

import os.path
try:
  import cPickle as pkl
except:
  import pickle as pkl
import nltk
from nltk import Tree

goeswith = pkl.load(open(os.path.abspath('dicts/en/goeswith.pkl')))
unstressed = pkl.load(open(os.path.abspath('dicts/en/unstressed.pkl')))
maybestressed = pkl.load(open(os.path.abspath('dicts/en/maybestressed.pkl')))

#***********************************************************************
class LexicalMetricalTree(Tree):

  #=====================================================================
  def __init__(self, node, children=None, lexical_stress=None):

    super(LexicalMetricalTree, self).__init__(node, children)
    self._cat = node
    self._label = None
    self._preterm = False
    self.set_lexical_stress(False)
    self.children = children

    if len(self) == 1 and isinstance(self[0], nltk.compat.string_types):
      self._preterm = True
      if lexical_stress is None:
        if   self[0].lower() in maybestressed:
          self.set_lexical_stress(None)
        elif self[0].lower() in unstressed or self[0].lower() in goeswith:
          self.set_lexical_stress(False)
        else:
          self.set_lexical_stress(True)
      else:
        self.set_lexical_stress(lexical_stress)
    else:
      for child in self:
        if child._lex_stress:
          self.set_lexical_stress(True)
          break
        elif child._lex_stress is None:
          self.set_lexical_stress(None)

  #=====================================================================
  def _get_last_preterm(self):

    if self._preterm:
      return self
    else:
      return self[-1]._get_last_preterm()

  #=====================================================================
  def _restress_last_branch(self):

    if self._preterm:
      if self[0].lower() in unstressed:
        self.set_lexical_stress(False)
      elif self[0].lower() in maybestressed:
        self.set_lexical_stress(None)
      else:
        self.set_lexical_stress(True)
    else:
      self[-1]._restress_last_branch()
      self._lex_stress = True
      for child in self:
        if child._lex_stress:
          self.set_lexical_stress(True)
        elif child._lex_stress is None and not self._lex_stress:
          self.set_lexical_stress(None)

  #=====================================================================
  def _pop_first_goeswith(self):

    if self._preterm:
      if self[0].lower() in goeswith:
        return self
      else:
        return None
    else:
      first_goeswith = self[0]._pop_first_goeswith()
      if self[0] == first_goeswith or len(self[0]) == 0:
        self.children.pop(0)
        self.pop(0)
      return first_goeswith

  #=====================================================================
  def contract(self):
    
    # Recurse
    for child in self:
      if isinstance(child, LexicalMetricalTree):
        child.contract()
    # Pull out adjacent tokens/goeswiths and merge them
    i = len(self) - 2
    while i >= 0:
      child = self[i]
      last_preterm = child._get_last_preterm()
      j = i + 1
      while j < len(self):
        next_child = self[j]
        first_goeswith = next_child._pop_first_goeswith()
        if first_goeswith is not None:
          # Merge their cats/leaves
          last_preterm._cat += '+'+first_goeswith._cat
          last_preterm[0] += first_goeswith[0]
          last_preterm.children[0] += first_goeswith.children[0]
          # Recalculate stressability based on new leaf
          child._restress_last_branch()
          # Disown empty children
          if len(next_child) == 0:
            self.pop(j)
          else:
            break
        else:
          break
      i -= 1
   
  #=====================================================================
  def generate_trees(self):

    if self._preterm:
      if self._lex_stress is not None:
        return [PhrasalMetricalTree(self._cat, self.children, self._lex_stress)]
      else:
        return [PhrasalMetricalTree(self._cat, self.children, True), PhrasalMetricalTree(self._cat, self.children, False)]
    else:
      possible_families = [[]]
      for child in self:
        possible_children = child.generate_trees()
        temp = []
        for possible_family in possible_families:
          for possible_child in possible_children:
            temp.append(possible_family + [possible_child])
        possible_families = temp

      temp = []
      for possible_family in possible_families:
        lex_stress = False
        for child in possible_family:
          if child._lex_stress == True:
            lex_stress = True
            break
          elif child._lex_stress is None:
            lex_stress = None
        temp.append(PhrasalMetricalTree(self._cat, possible_family, lex_stress))
      return temp

  #=====================================================================
  def set_lexical_stress(self, lexical_stress):
    
    # Override the default lexical_stress for prepositions
    if lexical_stress is None and self._preterm:
      if self._cat == 'IN':
        lexical_stress = False
      elif self._cat.startswith('R'):
        lexical_stress = True
      # TODO uncomment
      else:
        print self
    self._lex_stress = lexical_stress
    self._label = '%s/%s' % (self._cat, str(self._lex_stress))

  #=====================================================================
  def lexical_stress(self):

    return self._lex_stress

  #=====================================================================
  def preterminal(self):

    return self._preterm 

  #=====================================================================
  def category(self):

    return self._cat

  #=====================================================================
  def preterminals(self, leaves=True):

    if self._preterm:
      if leaves:
        yield self
      else:
        yield self._label
    for child in self:
      if isinstance(child, LexicalMetricalTree):
        for preterminal in child.preterminals(leaves=leaves):
          yield preterminal

  #=====================================================================
  @classmethod
  def convert(cls, tree):
    """
    Convert a tree between different subtypes of Tree.  ``cls`` determines
    which class will be used to encode the new tree.

    :type tree: Tree
    :param tree: The tree that should be converted.
    :return: The new Tree.
    """
    if isinstance(tree, Tree):
      children = [cls.convert(child) for child in tree]
      if isinstance(tree, LexicalMetricalTree):
        return cls(tree._cat, children, tree._lex_stress)
      else:
        return cls(tree._label, children)
    else:
      return tree

  #=====================================================================
  def copy(self, deep=False):

    if not deep:
      return type(self)(self._cat, self, lexical_stress=self._lex_stress)
    else:
      return type(self).convert(self)

#***********************************************************************
class PhrasalMetricalTree(LexicalMetricalTree):

  #=====================================================================
  def __init__(self, node, children=None, lexical_stress=None):

    super(PhrasalMetricalTree, self).__init__(node, children, lexical_stress)
    self.set_phrasal_stress(0)
    
    if not self._preterm:
      tempFlag = True
      # CSR
      if len(self) == 1:
        self[0]._stress = 0
      else:
        if self._cat.startswith('NP'):
          for i, child in enumerate(self[::-1]):
            if child._lex_stress and child._cat.startswith('NN'):
              if tempFlag == True:
                child.set_phrasal_stress(-1)
                tempFlag = False
              elif tempFlag == False:
                child.set_phrasal_stress(0)
                tempFlag = None
              elif tempFlag is None:
                child.set_phrasal_stress(-1)
        if not self._cat.startswith('NP') or not tempFlag:
          for i, child in enumerate(self[::-1]):
            if child._lex_stress:
              if tempFlag is not None:
                child.set_phrasal_stress(0)
                tempFlag = None
              else:
                child.set_phrasal_stress(-1)

  #=====================================================================
  def set_phrasal_stress(self, phrasal_stress):

    self._phr_stress = phrasal_stress
    if self._lex_stress:
      self._label = '%s/%s' % (self._cat, self._phr_stress)
    else:
      self._label = '%s/%s' % (self._cat, str(None))

  #=====================================================================
  def make_absolute(self, parent_phrasal_stress = 0):

    self.set_phrasal_stress(self._phr_stress + parent_phrasal_stress)
    if not self._preterm:
      for child in self:
        if isinstance(child, PhrasalMetricalTree):
          child.make_absolute(self._phr_stress)

#***********************************************************************
# 07/16: 1400 - 
if __name__ == '__main__':

  import sys
  import os
  import os.path
  from nltk.parse import stanford
  
  stanford_parser_dir = ''
  for directory in os.listdir(os.getcwd()):
    if directory.startswith('stanford-parser'):
      stanford_parser_dir = ''
  CLASSPATH = os.path.join(os.getcwd(), stanford_parser_dir)
  MODEL     = 'PCFG'
  XMX       = '1g'
  FILE      = None
  
  # $ python MetricalTree.py -c /home/tdozat/Scratch/Metrics/stanford-parser-full-2015-04-20/ -m RNN
  i = 0
  while i < len(sys.argv):
    if   sys.argv[i] == '-c' or sys.argv[i] == '--classpath':
      CLASSPATH = sys.argv.pop(i+1)
      # /home/tdozat/Scratch/Metrics/stanford-parser-full-2015-04-20/
    elif sys.argv[i] == '-m' or sys.argv[i] == '--model':
      MODEL = sys.argv.pop(i+1)
    elif sys.argv[i] == '-x' or sys.argv[i] == '--mx':
      XMX = sys.argv.pop(i+1)
    i+=1
  if 'CLASSPATH' not in os.environ:
    os.environ['CLASSPATH'] = os.path.pathsep.join([CLASSPATH+f for f in os.listdir(CLASSPATH)])
  elif CLASSPATH not in os.environ['CLASSPATH']:
    os.environ['CLASSPATH'] += os.path.pathsep.join([CLASSPATH+f for f in os.listdir(CLASSPATH)])
  
  parser = stanford.StanfordParser(model_path=os.path.join(os.path.normpath('edu/stanford/nlp/models/lexparser/english'+MODEL+'.ser.gz')), java_options='-mx%s'%XMX)
  ###
  # THIS LINE IS A HACK--should import "find_jar_iter" and find it that way
  parser._model_jar += os.path.pathsep+CLASSPATH+'ejml-0.23.jar'
  ###

  import re
  with open('Nixon.txt') as f:
    textfile = f.read()
  textfile = re.sub('\. ', '\.\n', textfile)
  textfile = re.sub(';', ';\n', textfile)
  textfile = re.sub(':', ':\n', textfile)
  textfile = re.sub('--', '--\n', textfile)
  sentences = parser.raw_parse_sents([textfile])
  stuff = []
  for line in sentences:
    for sentence in line:
      print sentence
      print 'Ambiguous preterminals:'
      sentence = LexicalMetricalTree.convert(sentence)
      sentence.contract()
      all_trees = sentence.generate_trees()
      print 'No. of sents:'
      print len(all_trees)
      stuff.append(len(all_trees))
      for tree in all_trees:
        tree = tree.copy(deep=True)
        tree.make_absolute()
        p = []
        for preterm in tree.preterminals():
          p.append('(%s %s)' % (preterm.label(), preterm[0]))
        #print ' '.join(p)

  import numpy as np
  print 'Average no. of sents: %.03f' % np.mean(stuff)
