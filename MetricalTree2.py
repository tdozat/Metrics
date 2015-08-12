#!/usr/bin/env python
#8/12/15: 1230-1345 1415-1615

import os.path
try:
  import cPickle as pkl
except:
  import pickle as pkl
import nltk
from nltk import Tree

#***********************************************************************
# Dependency-augmented syntactic tree class
class DepTree(Tree):
  
  #=====================================================================
  # Initialize
  def __init__(self, node, children=None, dep_label=None):
    
    super(DepTree, self).__init__(node, children)
    self._cat = node
    self._dep = dep_label
    self._label = None
    self._preterm = False
    self.children = children
    if len(self) == 1 and isinstance(self[0], nltk.compat.string_types):
      self._preterm = True

