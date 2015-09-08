#!/usr/bin/env python
#8/31/15: 1200-1600
#9/06/15: 1330-1945
#9/08/15: 11:30-12:15, 12:30-????

import os
from collections import defaultdict
import cPickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk import compat
from nltk.tree import Tree
import nltk.data
from deptree import DependencyTree, DependencyTreeParser

DATE = '2015-04-20'
MODELS_VERSION = '3.5.2'
EJML_VERSION = '0.23'
os.environ['STANFORD_PARSER'] = 'stanford-parser-full-%s/stanford-parser.jar' % DATE
os.environ['STANFORD_MODELS'] = 'stanford-parser-full-%s/stanford-parser-%s-models.jar' % (DATE, MODELS_VERSION)
os.environ['STANFORD_EJML']   = 'stanford-parser-full-%s/ejml-%s.jar' % (DATE, EJML_VERSION)
sylcmu = pkl.load(open('dicts/en/sylcmu.pkl'))
sent_splitter = nltk.data.load('tokenizers/punkt/english.pickle')

#***********************************************************************
# Split a text on certain punctuation
def pause_splitter(s):
  """"""
  
  s = s.strip()
  s = re.sub('([:;]|--+)', '\g<1>\n', s)
  s = s.split('\n')
  s = [sent for sents in s for sent in sent_splitter.tokenize(sents)]
  return s

#***********************************************************************
# Metrical Tree class
class MetricalTree(DependencyTree):
  """"""
  
  _unstressedWords = ('it',)
  _unstressedTags  = ('CC', 'PRP$', 'TO', 'UH', 'DT')
  _unstressedDeps  = ('det', 'expl', 'cc', 'mark')
  _ambiguousWords = ('this', 'that', 'these', 'those')
  _ambiguousTags  = ('MD', 'IN', 'PRP', 'WP$', 'PDT', 'WDT', 'WP', 'WRB')
  _ambiguousDeps  = ('cop', 'neg', 'aux', 'auxpass')
  _stressedWords = tuple()
  
  #=====================================================================
  # Initialize
  def __init__(self, node, children, dep=None, lstress=0, pstress=np.nan, stress=np.nan):
    """"""
    
    self._lstress = lstress
    self._pstress = pstress
    self._stress = stress
    super(MetricalTree, self).__init__(node, children, dep)
    self.set_label()
    if self._preterm:
      if self[0].lower() in sylcmu:
        syll_info = sylcmu[self[0].lower()]
        self._nseg = len(syll_info[0])
        self._nsyll = len(syll_info[1])
        self._nstress = len(filter(lambda x: x[1] in ('P', 'S'), syll_info[1]))
      else:
        self._nseg = np.nan
        self._nsyll = np.nan
        self._nstress = np.nan

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
  # Get the number of syllables
  def nseg(self):
    """"""
    
    return self._nseg
  
  #=====================================================================
  # Get the number of syllables
  def nsyll(self):
    """"""
    
    return self._nsyll
  
  #=====================================================================
  # Get the number of stresses
  def nstress(self):
    """"""
    
    return self._nstress
  
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
  # Get the number of syllables of the leaf nodes
  def nsylls(self, leaves=True):
    """"""
    
    for preterminal in self.preterminals(leaves=True):
      if leaves:
        yield (preterminal._nsyll, preterminal[0])
      else:
        yield preterminal._nsyll

  #=====================================================================
  # Set the lexical stress of the node
  def set_lstress(self):
    """"""
    
    if self._preterm:
      if self[0].lower() in super(MetricalTree, self)._contractables:
        self._lstress = np.nan
      elif self._cat in super(MetricalTree, self)._punctTags:
        self._lstress = np.nan
      
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
          elif not np.isnan(child._pstress):
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
  def disambiguate(self, syll=False):
    """"""
    
    if self._preterm:
      if self._lstress != -.5:
        return [self.copy()]
      else:
        alts = []
        if not syll or np.isnan(self._nsyll) or self._nsyll == 2:
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
# Parser for Metrical Trees
class MetricalTreeParser:
  """"""
  
  #=====================================================================
  # Initialize
  def __init__(self, deptreeParser=None):
    """"""
    
    if deptreeParser is None:
      sys.stderr.write('No deptreeParser provided, defaulting to PCFG\n')
      deptreeParser = 'PCFG'
    if isinstance(deptreeParser, compat.string_types):
      deptreeParser = DependencyTreeParser(model_path='stanford-parser-full-%s/edu/stanford/nlp/models/lexparser/english%s.ser.gz' % (DATE, deptreeParser))
    elif not isinstance(deptreeParser, DependencyTreeParser):
      raise ValueError('Provided an invalid dependency tree parser')
    self.deptreeParser = deptreeParser
  
  #=====================================================================
  # Use StanfordParser to parse a list of tokens
  def dep_parse_sents(self, sentences, verbose=False):
    """"""
    
    return self.deptreeParser.parse_sents(sentences, verbose)
  
  #=====================================================================
  # Use StanfordParser to parse a raw sentence
  def dep_raw_parse(self, sentence, verbose=False):
    """"""
    
    return self.deptreeParser.raw_parse(sentence, verbose)
  
  #=====================================================================
  # Use StanfordParser to parse multiple raw sentences
  def dep_raw_parse_sents(self, sentences, verbose=False):
    """"""
    
    return self.deptreeParser.raw_parse_sents(sentences, verbose)
  
  #=====================================================================
  # Use StanfordParser to parse multiple preprocessed sentences
  def dep_tagged_parse_sent(self, sentence, verbose=False):
    """"""
    
    return self.deptreeParser.tagged_parse_sent(sentence, verbose)
  
  #=====================================================================
  # Use StanfordParser to parse multiple preprocessed sentences
  def dep_tagged_parse_sents(self, sentences, verbose=False):
    """"""
    
    return self.deptreeParser.tagged_parse_sents(sentences, verbose)
 
  #=====================================================================
  # Parse a list of tokens into lexical Metrical Trees
  def lex_parse_sents(self, sentences, verbose=False):
    """"""
    
    sentences = self.dep_parse_sents(sentences, verbose)
    for tree in sentences:
      for t in tree:
        t = MetricalTree.convert(t)
        t.set_lstress()
        yield t
  
  #=====================================================================
  # Parse a raw sentence into lexical Metrical Trees
  def lex_raw_parse(self, sentence, verbose=False):
    """"""
    
    sentence = self.dep_raw_parse(sentence, verbose)
    for t in sentence:
      t = MetricalTree.convert(t)
      t.set_lstress()
      yield t
  
  #=====================================================================
  # Parse a string into lexical Metrical Trees
  def lex_raw_parse_sents(self, sentences, verbose=False):
    """"""
    
    sentences = self.dep_raw_parse_sents(sentences, verbose)
    for tree in sentences:
      for t in tree:
        t = MetricalTree.convert(t)
        t.set_lstress()
        yield t
  
  #=====================================================================
  # Parse a tagged sentence into lexical Metrical Trees
  def lex_tagged_parse(self, sentence, verbose=False):
    """"""
    
    sentence = self.dep_tagged_parse(sentence, verbose)
    for t in sentence:
      t = MetricalTree.convert(t)
      t.set_lstress()
      yield t
  
  #=====================================================================
  # Parse a raw sentence into lexical Metrical Trees
  def lex_tagged_parse_sents(self, sentences, verbose=False):
    """"""
    
    sentences = self.dep_tagged_parse_sents(sentences, verbose)
    for tree in sentences:
      for t in tree:
        t = MetricalTree.convert(t)
        t.set_lstress()
        yield t
  
  #=====================================================================
  # Parse a list of tokens into phrasal Metrical Trees
  def phr_parse_sents(self, sentences, syll=False, verbose=True):
    """"""
    
    for t in self.lex_parse_sents(sentences, verbose):
      trees = t.disambiguate(syll)
      for tree in trees:
        tree.set_pstress()
        tree.set_stress()
      yield trees
  
  #=====================================================================
  # Parse a string into phrasal Metrical Trees
  def phr_raw_parse(self, sentences, syll=False, verbose=True):
    """"""
    
    for t in self.lex_raw_parse(sentences, verbose):
      trees = t.disambiguate(syll)
      for tree in trees:
        tree.set_pstress()
        tree.set_stress()
      yield trees
      
  #=====================================================================
  # Parse a list of strings into phrasal Metrical Trees
  def phr_raw_parse_sents(self, sentences, syll=False, verbose=True):
    """"""
    
    for t in self.lex_raw_parse_sents(sentences, verbose):
      trees = t.disambiguate(syll)
      for tree in trees:
        tree.set_pstress()
        tree.set_stress()
      yield trees
  
  #=====================================================================
  # Parse a list of tagged strings into phrasal Metrical Trees
  def phr_tagged_parse(self, sentences, syll=False, verbose=True):
    """"""
    
    for t in self.lex_tagged_parse(sentences, verbose):
      trees = t.disambiguate(syll)
      for tree in trees:
        tree.set_pstress()
        tree.set_stress()
      yield trees
  
  #=====================================================================
  # Parse a list of strings into phrasal Metrical Trees
  def phr_tagged_parse_sents(self, sentences, syll=False, verbose=True):
    """"""
    
    for t in self.lex_tagged_parse_sents(sentences, verbose):
      trees = t.disambiguate(syll)
      for tree in trees:
        tree.set_pstress()
        tree.set_stress()
      yield trees
  
  #=====================================================================
  # Parse a list of tokens into phrasal Metrical Trees
  def stats_parse_sents(self, sentences, arto=False, verbose=True):
    """"""
    
    data = defaultdict(list)
    i = 0
    for t in self.lex_parse_sents(sentences, verbose):
      i += 1
      trees1 = t.disambiguate(syll=True)
      trees2 = t.disambiguate(syll=False)
      for tree1, tree2 in zip(trees1, trees2):
        tree1.set_pstress()
        tree2.set_pstress()
        tree1.set_stress()
        tree2.set_stress()
      j = 0
      preterms1a = trees1[0].preterminals()
      preterms1b = trees1[-1].preterminals()
      preterms2a = trees2[0].preterminals()
      preterms2b = trees2[-1].preterminals()
      sent = ' '.join([preterm[0] for preterm in preterms1a])
      for preterm1a, preterm1b, preterm2a, preterm2b in zip(preterms1a, preterms1b, preterms2a, preterms2b):
        j += 1
        data['widx'].append(j)
        data['word'].append(preterm1a[0])
        data['nseg'].append(preterm1a.nseg())
        data['nsyll'].append(preterm1a.nsyll())
        data['nstress'].append(preterm1a.nstress())
        data['pos'].append(preterm1a.category())
        data['dep'].append(preterm1a.dependency())
        if arto:
          data['m1a'].append(-(preterm1a.stress()-1))
          data['m1b'].append(-(preterm1b.stress()-1))
          data['m2a'].append(-(preterm2a.stress()-1))
          data['m2b'].append(-(preterm2b.stress()-1))
        else:
          data['m1a'].append(preterm1a.stress())
          data['m1b'].append(preterm1b.stress())
          data['m2a'].append(preterm2a.stress())
          data['m2b'].append(preterm2b.stress())
        data['mmean'].append(np.mean([preterm1a.stress(), preterm1b.stress(), preterm2a.stress(), preterm2b.stress()]))
        data['sidx'].append(i)
        data['sent'].append(sent)
        data['log2(nparse1)'].append(np.log2(len(preterms1a)))
        data['log2(nparse2)'].append(np.log2(len(preterms2a)))
    for k, v in data:
      data[k] = pd.Series(data[v])
    return pd.DataFrame(data, columns=['widx', 'word', 'nseg', 'nsyll', 'nstress',
                                       'pos', 'dep',
                                       'm1a', 'm1b', 'm2a', 'm2b', 'mmean',
                                       'sidx', 'sent', 'log2(nparse1)', 'log2(nparse2)'])
  
  #=====================================================================
  # Parse a string into phrasal Metrical Trees
  def stats_raw_parse(self, sentence, arto=False, verbose=True):
    """"""
    
    data = defaultdict(list)
    i = 0
    for t in self.lex_raw_parse(sentence, verbose):
      i += 1
      trees1 = t.disambiguate(syll=True)
      trees2 = t.disambiguate(syll=False)
      for tree1, tree2 in zip(trees1, trees2):
        tree1.set_pstress()
        tree2.set_pstress()
        tree1.set_stress()
        tree2.set_stress()
      j = 0
      preterms1a = trees1[0].preterminals()
      preterms1b = trees1[-1].preterminals()
      preterms2a = trees2[0].preterminals()
      preterms2b = trees2[-1].preterminals()
      sent = ' '.join([preterm[0] for preterm in preterms1a])
      for preterm1a, preterm1b, preterm2a, preterm2b in zip(preterms1a, preterms1b, preterms2a, preterms2b):
        j += 1
        data['widx'].append(j)
        data['word'].append(preterm1a[0])
        data['nseg'].append(preterm1a.nseg())
        data['nsyll'].append(preterm1a.nsyll())
        data['nstress'].append(preterm1a.nstress())
        data['pos'].append(preterm1a.category())
        data['dep'].append(preterm1a.dependency())
        if arto:
          data['m1a'].append(-(preterm1a.stress()-1))
          data['m1b'].append(-(preterm1b.stress()-1))
          data['m2a'].append(-(preterm2a.stress()-1))
          data['m2b'].append(-(preterm2b.stress()-1))
        else:
          data['m1a'].append(preterm1a.stress())
          data['m1b'].append(preterm1b.stress())
          data['m2a'].append(preterm2a.stress())
          data['m2b'].append(preterm2b.stress())
        data['mmean'].append(np.mean([preterm1a.stress(), preterm1b.stress(), preterm2a.stress(), preterm2b.stress()]))
        data['sidx'].append(i)
        data['sent'].append(sent)
        data['log2(nparse1)'].append(np.log2(len(preterms1a)))
        data['log2(nparse2)'].append(np.log2(len(preterms2a)))
    for k, v in data:
      data[k] = pd.Series(data[v])
    return pd.DataFrame(data, columns=['widx', 'word', 'nseg', 'nsyll', 'nstress',
                                       'pos', 'dep',
                                       'm1a', 'm1b', 'm2a', 'm2b', 'mmean',
                                       'sidx', 'sent', 'log2(nparse1)', 'log2(nparse2)'])
  
  #=====================================================================
  # Parse a string into phrasal Metrical Trees
  def stats_raw_parse_sents(self, sentences, arto=False, verbose=True):
    """"""
    
    data = defaultdict(list)
    i = 0
    for t in self.lex_raw_parse_sents(sentences, verbose):
      i += 1
      trees1 = t.disambiguate(syll=True)
      trees2 = t.disambiguate(syll=False)
      for tree1, tree2 in zip(trees1, trees2):
        tree1.set_pstress()
        tree2.set_pstress()
        tree1.set_stress()
        tree2.set_stress()
      j = 0
      preterms1a = trees1[0].preterminals()
      preterms1b = trees1[-1].preterminals()
      preterms2a = trees2[0].preterminals()
      preterms2b = trees2[-1].preterminals()
      sent = ' '.join([preterm[0] for preterm in trees1[0].preterminals()])
      for preterm1a, preterm1b, preterm2a, preterm2b in zip(preterms1a, preterms1b, preterms2a, preterms2b):
        j += 1
        data['widx'].append(j)
        data['word'].append(preterm1a[0])
        data['nseg'].append(preterm1a.nseg())
        data['nsyll'].append(preterm1a.nsyll())
        data['nstress'].append(preterm1a.nstress())
        data['pos'].append(preterm1a.category())
        data['dep'].append(preterm1a.dependency())
        if arto:
          data['m1a'].append(-(preterm1a.stress()-1))
          data['m1b'].append(-(preterm1b.stress()-1))
          data['m2a'].append(-(preterm2a.stress()-1))
          data['m2b'].append(-(preterm2b.stress()-1))
          data['mmean'].append(-(np.mean([preterm1a.stress(), preterm1b.stress(), preterm2a.stress(), preterm2b.stress()])-1))
        else:
          data['m1a'].append(preterm1a.stress())
          data['m1b'].append(preterm1b.stress())
          data['m2a'].append(preterm2a.stress())
          data['m2b'].append(preterm2b.stress())
          data['mmean'].append(np.mean([preterm1a.stress(), preterm1b.stress(), preterm2a.stress(), preterm2b.stress()]))
        data['sidx'].append(i)
        data['sent'].append(sent)
        data['log2(nparse1)'].append(np.log2(len(trees1)))
        data['log2(nparse2)'].append(np.log2(len(trees2)))
    return pd.DataFrame(data, columns=['widx', 'word', 'nseg', 'nsyll', 'nstress',
                                       'pos', 'dep',
                                       'm1a', 'm1b', 'm2a', 'm2b', 'mmean',
                                       'sidx', 'sent', 'log2(nparse1)', 'log2(nparse2)'])
  
  #=====================================================================
  # Parse a list of tagged tokens into phrasal Metrical Trees
  def stats_tagged_parse(self, sentence, arto=False, verbose=True):
    """"""
    
    data = defaultdict(list)
    i = 0
    for t in self.lex_tagged_parse(sentence, verbose):
      i += 1
      trees1 = t.disambiguate(syll=True)
      trees2 = t.disambiguate(syll=False)
      for tree1, tree2 in zip(trees1, trees2):
        tree1.set_pstress()
        tree2.set_pstress()
        tree1.set_stress()
        tree2.set_stress()
      j = 0
      preterms1a = trees1[0].preterminals()
      preterms1b = trees1[-1].preterminals()
      preterms2a = trees2[0].preterminals()
      preterms2b = trees2[-1].preterminals()
      sent = ' '.join([preterm[0] for preterm in preterms1a])
      for preterm1a, preterm1b, preterm2a, preterm2b in zip(preterms1a, preterms1b, preterms2a, preterms2b):
        j += 1
        data['widx'].append(j)
        data['word'].append(preterm1a[0])
        data['nseg'].append(preterm1a.nseg())
        data['nsyll'].append(preterm1a.nsyll())
        data['nstress'].append(preterm1a.nstress())
        data['pos'].append(preterm1a.category())
        data['dep'].append(preterm1a.dependency())
        if arto:
          data['m1a'].append(-(preterm1a.stress()-1))
          data['m1b'].append(-(preterm1b.stress()-1))
          data['m2a'].append(-(preterm2a.stress()-1))
          data['m2b'].append(-(preterm2b.stress()-1))
        else:
          data['m1a'].append(preterm1a.stress())
          data['m1b'].append(preterm1b.stress())
          data['m2a'].append(preterm2a.stress())
          data['m2b'].append(preterm2b.stress())
        data['mmean'].append(np.mean([preterm1a.stress(), preterm1b.stress(), preterm2a.stress(), preterm2b.stress()]))
        data['sidx'].append(i)
        data['sent'].append(sent)
        data['log2(nparse1)'].append(np.log2(len(preterms1a)))
        data['log2(nparse2)'].append(np.log2(len(preterms2a)))
    for k, v in data:
      data[k] = pd.Series(data[v])
    return pd.DataFrame(data, columns=['widx', 'word', 'nseg', 'nsyll', 'nstress',
                                       'pos', 'dep',
                                       'm1a', 'm1b', 'm2a', 'm2b', 'mmean',
                                       'sidx', 'sent', 'log2(nparse1)', 'log2(nparse2)'])
  
  #=====================================================================
  # Parse a list of tagged tokens into phrasal Metrical Trees
  def stats_tagged_parse_sents(self, sentences, arto=False, verbose=True):
    """"""
    
    data = defaultdict(list)
    i = 0
    for t in self.lex_tagged_parse_sents(sentence, verbose):
      i += 1
      trees1 = t.disambiguate(syll=True)
      trees2 = t.disambiguate(syll=False)
      for tree1, tree2 in zip(trees1, trees2):
        tree1.set_pstress()
        tree2.set_pstress()
        tree1.set_stress()
        tree2.set_stress()
      j = 0
      preterms1a = trees1[0].preterminals()
      preterms1b = trees1[-1].preterminals()
      preterms2a = trees2[0].preterminals()
      preterms2b = trees2[-1].preterminals()
      sent = ' '.join([preterm[0] for preterm in preterms1a])
      for preterm1a, preterm1b, preterm2a, preterm2b in zip(preterms1a, preterms1b, preterms2a, preterms2b):
        j += 1
        data['widx'].append(j)
        data['word'].append(preterm1a[0])
        data['nseg'].append(preterm1a.nseg())
        data['nsyll'].append(preterm1a.nsyll())
        data['nstress'].append(preterm1a.nstress())
        data['pos'].append(preterm1a.category())
        data['dep'].append(preterm1a.dependency())
        if arto:
          data['m1a'].append(-(preterm1a.stress()-1))
          data['m1b'].append(-(preterm1b.stress()-1))
          data['m2a'].append(-(preterm2a.stress()-1))
          data['m2b'].append(-(preterm2b.stress()-1))
        else:
          data['m1a'].append(preterm1a.stress())
          data['m1b'].append(preterm1b.stress())
          data['m2a'].append(preterm2a.stress())
          data['m2b'].append(preterm2b.stress())
        data['mmean'].append(np.mean([preterm1a.stress(), preterm1b.stress(), preterm2a.stress(), preterm2b.stress()]))
        data['sidx'].append(i)
        data['sent'].append(sent)
        data['log2(nparse1)'].append(np.log2(len(preterms1a)))
        data['log2(nparse2)'].append(np.log2(len(preterms2a)))
    for k, v in data:
      data[k] = pd.Series(data[v])
    return pd.DataFrame(data, columns=['widx', 'word', 'nseg', 'nsyll', 'nstress',
                                       'pos', 'dep',
                                       'm1a', 'm1b', 'm2a', 'm2b', 'mmean',
                                       'sidx', 'sent', 'log2(nparse1)', 'log2(nparse2)'])
  
#***********************************************************************
# Test the module
if __name__ == '__main__':
  """"""
  
  import os
  import re
  import time
  parser = DependencyTreeParser(model_path='stanford-parser-full-%s/edu/stanford/nlp/models/lexparser/englishRNN.ser.gz' % DATE)
  parser = MetricalTreeParser(parser)
  
  sents = []
  print '=== The Hobbit Opening ==='
  with open('Hobbit.txt') as f:
    for line in f:
      sents.extend(pause_splitter(line))
  df = parser.stats_raw_parse_sents(sents, arto=True)
  df.to_csv(open('Hobbit.csv', 'w'), index_label=False)
  #forests = parser.phr_raw_parse_sents(sents)
  #for forest in forests:
  #  for stress1, stress2 in zip(forest[0].stresses(arto=True), forest[-1].stresses(arto=True)):
  #    print '%s %s %s' % (str(stress1[0]).ljust(4), str(stress2[0]).ljust(4), stress1[1])
  #  print 'no. of parses: %d\n' % len(forest)
  sents = []
  print '\n=== The Nixon Inauguration ==='
  with open('Nixon.txt') as f:
    for line in f:
      sents.extend(pause_splitter(line))
  df = parser.stats_raw_parse_sents(sents, arto=True)
  df.to_csv(open('Nixon.csv', 'w'), index_label=False)
  #forests = parser.phr_raw_parse_sents(sents)
  #for forest in forests:
  #  for stress1, stress2 in zip(forest[0].stresses(arto=True), forest[-1].stresses(arto=True)):
  #    print '%s %s %s' % (str(stress1[0]).ljust(4), str(stress2[0]).ljust(4), stress1[1])
  #  print 'no. of parses: %d\n' % len(forest)
  print 'Works!'