#!/usr/bin/env python
#8/31/15: 1200-1600

import os
import nltk
from nltk.tree import Tree
from deptree import DependencyTree, DependencyTreeParser, punct_split

DATE = '2015-04-20'
MODELS_VERSION = '3.5.2'
EJML_VERSION = '0.23'
os.environ['STANFORD_PARSER'] = 'stanford-parser-full-%s/stanford-parser.jar' % DATE
os.environ['STANFORD_MODELS'] = 'stanford-parser-full-%s/stanford-parser-%s-models.jar' % (DATE, MODELS_VERSION)
os.environ['STANFORD_EJML']   = 'stanford-parser-full-%s/ejml-%s.jar' % (DATE, EJML_VERSION)


#***********************************************************************
# Metrical Tree class
class MetricalTree(DependencyTree):
  """
    Step 1: Parse the string into a DependencyTree using the DependencyTreeParser from the deptree module
    Step 2: Convert the DependencyTree into a MetricalTree
    Step 3: Set the lexical stress
    Step 4: Disambiguate 
    Step 6: Set the phrasal stress
    Step 7: Set the total stress
    Example:
      >>> parser = DependencyTreeParser()
      >>> sent_dtree = parser.raw_parse(sent_str)
      >>> for tree in sent_dtree:
      ...   sent_mtree = MetricalTree.convert(t)
      ...   sent_mtree.set_lstress()
      ...   sent_mtrees = sent_mtree.disambiguate()
      ...   for i, sent_mtree in enumerate(sent_mtrees):
      ...     sent_mtrees[i] = sent_mtree.copy(deep=True)
      ...     sent_mtree.set_stress()
  """
  
  _unstressedWords = ('it',)
  _unstressedTags  = ('CC', 'PRP$', 'TO', 'UH', 'DT')
  _unstressedDeps  = ('det', 'expl', 'cc', 'mark')
  _ambiguousWords = ('this', 'that', 'these', 'those')
  _ambiguousTags  = ('MD', 'IN', 'PRP', 'WP$', 'PDT', 'WDT', 'WP', 'WRB')
  _ambiguousDeps  = ('cop', 'neg', 'aux', 'auxpass')
  _stressedWords = tuple()
  
  #=====================================================================
  # Initialize
  def __init__(self, node, children, dep=None, lstress=0, pstress=None, stress=None):
    """"""
    
    self._lstress = lstress
    self._pstress = pstress
    self._stress = stress
    super(MetricalTree, self).__init__(node, children, dep)
    self.set_label()
    
  #=====================================================================
  # Get the lexical stress of the node
  def lstress(self):
    """"""
    
    return self._lstress
  
  #=====================================================================
  # Get the phrasal stress of the node
  def pstress(self):
    """"""
    
    return self._pstress
  
  #=====================================================================
  # Get the stress of the node
  def stress(self):
    """"""
    
    return self._stress
  
  #=====================================================================
  # Get the lexical stress of the leaf nodes
  def lstresses(self, leaves=True):
    """"""
    
    for preterminal in self.preterminals(leaves=True):
      if leaves:
        yield (preterminal._lstress, preterminal[0])
      else:
        yield preterminal._lstress

  #=====================================================================
  # Get the phrasal stress of the leaf nodes
  def pstresses(self, leaves=True):
    """"""
    
    for preterminal in self.preterminals(leaves=True):
      if leaves:
        yield (preterminal._pstress, preterminal[0])
      else:
        yield preterminal._pstress

  #=====================================================================
  # Get the lexical stress of the leaf nodes
  def stresses(self, leaves=True, arto=False):
    """"""
    
    for preterminal in self.preterminals(leaves=True):
      if leaves:
        if arto:
          if preterminal._stress is None:
            yield (None, preterminal[0])
          elif preterminal._lstress == -1:
            yield (0, preterminal[0])
          else:
            yield (-(preterminal._stress-1), preterminal[0])
        else:
          yield (preterminal._stress, preterminal[0])
      else:
        if arto:
          if preterminal._stress is None:
            yield None
          elif preterminal._lstress == -1:
            yield 0
          else:
            yield -(preterminal._stress-1)
        else:
          yield preterminal._stress

  #=====================================================================
  # Set the lexical stress of the node
  def set_lstress(self):
    """"""
    
    if self._preterm:
      if self[0].lower() in super(MetricalTree, self)._contractables:
        self._lstress = None
      elif self._cat in super(MetricalTree, self)._punctTags:
        self._lstress = None
      
      elif self[0].lower() in MetricalTree._unstressedWords:
        self._lstress = -1
      elif self[0].lower() in MetricalTree._ambiguousWords:
        self._lstress = -.5
      elif self[0].lower() in MetricalTree._stressedWords:
        self._lstress = 0
        
      elif self._cat in MetricalTree._unstressedTags:
        self._lstress = -1
      elif self._cat in MetricalTree._ambiguousTags:
        self._lstress = -.5
        
      elif self._dep in MetricalTree._unstressedDeps:
        self._lstress = -1
      elif self._dep in MetricalTree._ambiguousDeps:
        self._lstress = -.5
        
      else:
        self._lstress = 0
      
      if self[0].lower() == 'that' and (self._cat == 'DT' or self._dep == 'det'):
        self._lstress = -.5
    else:
      for child in self:
        child.set_lstress()
    self.set_label()
    
  #=====================================================================
  # Set the phrasal stress of the tree
  def set_pstress(self):
    """"""
    
    # Basis
    if self._preterm:
      try: assert self._lstress != -.5
      except: raise ValueError('The tree must be disambiguated before assigning phrasal stress')
      self._pstress = self._lstress
    else:
      # Recurse
      for child in self:
        child.set_pstress()
      assigned = False
      # Noun compounds (look for sequences of N*)
      if self._cat == 'NP':
        skipIdx = None
        i = len(self)
        for child in self[::-1]:
          i -= 1
          if child._cat.startswith('NN'):
            if not assigned and skipIdx is None:
              skipIdx = i
              child._pstress = -1
              child.set_label()
            elif not assigned:
              child._pstress = 0
              child.set_label()
              assigned = True
            else:
              child._pstress = -1
              child.set_label()
          elif assigned:
            child._pstress = -1
            child.set_label()
          else:
            if not assigned and skipIdx is not None:
              self[skipIdx]._pstress = 0
              self[skipIdx].set_label()
              assigned = True
              child._pstress = -1
              child.set_label()
            else:
              break
        if not assigned and skipIdx is not None:
          self[skipIdx]._pstress = 0
          self[skipIdx].set_label()
          assigned = True
      # Everything else
      if not assigned:
        for child in self[::-1]:
          if not assigned and child._pstress == 0:
            assigned = True
          elif child._pstress is not None:
            child._pstress = -1
            child.set_label()
      if not assigned:
        self._pstress = -1
      else:
        self._pstress = 0
    self.set_label()
    
  #=====================================================================
  # Set the total of the tree
  def set_stress(self, stress=0):
    """"""
    
    # Basis
    if self._lstress is not None and self._pstress is not None:
      self._stress = self._lstress + self._pstress + stress
    if not self._preterm:
      for child in self:
        child.set_stress(self._stress)
    self.set_label()
    
  #=====================================================================
  # Reset the label of the node (cat < dep < lstress < pstress < stress
  def set_label(self):
    """"""
    
    if self._stress is not None:
      self._label = '%s/%s' % (self._cat, self._stress)
    elif self._pstress is not None:
      self._label = '%s/%s' % (self._cat, self._pstress)
    elif self._lstress is not None:
      self._label = '%s/%s' % (self._cat, self._lstress)
    elif self._dep is not None:
      self._label = '%s/%s' % (self._cat, self._dep)
    else:
      self._label = '%s' % self._cat
  
  #=====================================================================
  # Convert between different subtypes of Metrical Trees
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
      if isinstance(tree, MetricalTree):
        return cls(tree._cat, children, tree._dep, tree._lstress)
      elif isinstance(tree, DependencyTree):
        return cls(tree._cat, children, tree._dep)
      else:
        return cls(tree._label, children)
    else:
      return tree

  #=====================================================================
  # Lexically disambiguate an ambiguous Metrical Tree
  def disambiguate(self):
    """"""
    
    if self._preterm:
      if self._lstress != -.5:
        return [self.copy()]
      else:
        alts = []
        self._lstress = -1
        alts.append(self.copy())
        self._lstress = 0
        alts.append(self.copy())
        self._lstress = -.5
        return alts
    else:
      alts = [[]]
      for child in self:
        child_alts = child.disambiguate()
        for i in xrange(len(alts)):
          alt = alts.pop(0)
          for child_alt in child_alts:
            alts.append(alt + [child_alt])
      return [MetricalTree(self._cat, alt, self._dep) for alt in alts]
  
  #=====================================================================
  # Copy the tree
  def copy(self, deep=False):
    """"""

    if not deep:
      return type(self)(self._cat, self, dep=self._dep, lstress=self._lstress)
    else:
      return type(self).convert(self)
  
#***********************************************************************
# Test the module
if __name__ == '__main__':
  """"""
  
  import os
  import re
  import time
  parser = DependencyTreeParser(model_path='stanford-parser-full-%s/edu/stanford/nlp/models/lexparser/englishRNN.ser.gz' % DATE)
  sents = []
  print '=== The Hobbit Opening ==='
  with open('Hobbit.txt') as f:
    for line in f:
      sents.extend(punct_split(line))
  sentences = parser.raw_parse_sents(sents)
  start = time.time()
  for tree in sentences:
    for t in tree:
      t = MetricalTree.convert(t)
      t.set_lstress()
      print t
      ts = t.disambiguate()
      for t in ts:
        t.set_pstress()
        t.set_stress()
      for stress1, stress2 in zip(ts[0].stresses(arto=True), ts[-1].stresses(arto=True)):
        print '%s %s %s' % (str(stress1[0]).ljust(4), str(stress2[0]).ljust(4), stress1[1])
      print 'no. of parses: %d\n' % len(ts)
  sents = []
  print '\n=== The Nixon Inauguration ==='
  with open('Nixon.txt') as f:
    for line in f:
      sents.extend(punct_split(line))
  sentences = parser.raw_parse_sents(sents)
  for tree in sentences:
    for t in tree:
      t = MetricalTree.convert(t)
      t.set_lstress()
      print t
      ts = t.disambiguate()
      for t in ts:
        t.set_pstress()
        t.set_stress()
      for stress1, stress2 in zip(ts[0].stresses(arto=True), ts[-1].stresses(arto=True)):
        print '%s %s %s' % (str(stress1[0]).ljust(4), str(stress2[0]).ljust(4), stress1[1])
      print 'no. of parses: %d\n' % len(ts)
  print 'Works!'