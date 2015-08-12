#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Natural Language Toolkit: Interface to the Stanford Dependency Parser
#
# Copyright (C) 2015 Tim Dozat
# Author: Tim Dozat <tdozat@stanford.edu>
# Author of the Stanford Parser nltk code: Steven Xu <xxu@student.unimelb.edu.au>
#
# For license information, see LICENSE.TXT

from __future__ import unicode_literals

import tempfile
import os
import re
from subprocess import PIPE

from nltk import compat
from nltk.internals import find_jar, find_jar_iter, config_java, java, _java_options

from nltk.parse.api import ParserI
from nltk.parse import DependencyGraph

_stanford_url = 'http://nlp.stanford.edu/software/lex-parser.shtml'

#***********************************************************************
# Interface to the Stanford Dependency Parser
class StanfordDependencyParser(ParserI):
  """"""
  
  _MODEL_JAR_PATTERN = r'stanford-parser-(\d+)(\.(\d+))+-models\.jar'
  _JAR = 'stanford-parser.jar'
  
  #=====================================================================
  # Initialize
  def __init__(self,  path_to_jar=None, path_to_models_jar=None,
      model_path='edu/stanford/nlp/models/parser/nndep/english_UD.gz',
      encoding='utf8', verbose=False, java_options='-mx1024m'):
    """"""
    
    self._stanford_jar = find_jar(
      self._JAR, path_to_jar,
      env_vars=('STANFORD_PARSER',),
      searchpath=(), url=_stanford_url,
      verbose=verbose)
    
    # find the most recent model
    self._model_jar=max(
      find_jar_iter(
        self._MODEL_JAR_PATTERN, path_to_models_jar,
        env_vars=('STANFORD_MODELS',),
        searchpath=(), url=_stanford_url,
        verbose=verbose, is_regex=True),
      key=lambda model_name: re.match(self._MODEL_JAR_PATTERN, model_name))
      
    self.model_path = model_path
    self._encoding = encoding
    self.java_options = java_options
    
  #=====================================================================
  # Parse the output
  @staticmethod
  def _parse_trees_output(output_):
    """"""
    
    res = []
    cur_lines = []
    for line in output_.splitlines(False):
      if line == '':
        # TODO update this part
        res.append(iter([DependencyGraph.fromstring('\n'.join(cur_lines))]))
        cur_lines = []
      else:
        cur_lines.append(line)
    return iter(res)
  
  #=====================================================================
  # Use StanfordParser to parse multiple preprocessed sentences
  def parse_sents(self, sentences, verbose=False):
    """"""
    
    cmd = [
      'edu.stanford.nlp.parser.nndep.english_UD',
      '-model', self.model_path,
      '-sentences', 'newline',
      '-outputformat', 'penn',
      '-tokenized',
      '-escaper', 'edu.stanford.nlp.process.PTBEscapingProcessor',
    ]
    return self._parse_trees_output(self._execute(
      cmd, '\n'.join(' '.join(sentence) for sentence in sentences), verbose))

  #=====================================================================
  # Use StanfordParser to parse a raw sentence.
  def raw_parse(self, sentence, verbose=False):
    """"""
    
    return next(self.raw_parse_sents([sentence], verbose))

  #=====================================================================
  # Use StanfordParser to parse raw sentences.
  def raw_parse_sents(self, sentences, verbose=False):
    """"""
    
    cmd = [
      'edu.stanford.nlp.parser.nndep.english_UD',
      '-model', self.model_path,
      '-sentences', 'newline',
      '-outputFormat', 'penn',
    ]
    return self._parse_trees_output(self._execute(cmd, '\n'.join(sentences), verbose))
  
  #=====================================================================
  # Use StanfordParser to parse raw sentences.
  def tagged_parse(self, sentence, verbose=False):
    """"""
    
    return next(self.tagged_parse_sents([sentence], verbose))
  
  #=====================================================================
  # Use StanfordParser to parse raw sentences.
  def tagged_parse_sents(self, sentences, verbose=False):
    """"""
    
    tag_separator = '/'
    cmd = [
      'edu.stanford.nlp.parser.lexparser.LexicalizedParser',
      '-model', self.model_path,
      '-sentences', 'newline',
      '-outputFormat', 'penn',
      '-tokenized',
      '-tagSeparator', tag_separator,
      '-tokenizerFactory', 'edu.stanford.nlp.process.WhitespaceTokenizer',
      '-tokenizerMethod', 'newCoreLabelTokenizerFactory',
    ]
    # We don't need to escape slashes as "splitting is done on the last instance of the character in the token"
    return self._parse_trees_output(self._execute(
      cmd, '\n'.join(' '.join(tag_separator.join(tagged) for tagged in sentence) for sentence in sentences), verbose))
  
  #=====================================================================
  # Execute
  def _execute(self, cmd, input_, verbose=False):
    """"""
    
    encoding = self._encoding
    cmd.extend(['-encoding', encoding])

    default_options = ' '.join(_java_options)

    # Configure java.
    config_java(options=self.java_options, verbose=verbose)

    # Windows is incompatible with NamedTemporaryFile() without passing in delete=False.
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as input_file:
      # Write the actual sentences to the temporary input file
      if isinstance(input_, compat.text_type) and encoding:
          input_ = input_.encode(encoding)
      input_file.write(input_)
      input_file.flush()

      cmd.append(input_file.name)

      # Run the tagger and get the output.
      stdout, stderr = java(cmd, classpath=(self._stanford_jar, self._model_jar),
                            stdout=PIPE, stderr=PIPE)
      stdout = stdout.decode(encoding)

    os.unlink(input_file.name)

    # Return java configurations to their default values.
    config_java(options=default_options, verbose=False)

    return stdout
  
#***********************************************************************
# Set up the module
def setup_module(module):
  """"""
  
  from nose import SkipTest

  try:
    StanfordParser(
      model_path='edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'
    )
  except LookupError:
    raise SkipTest('doctests from nltk.parse.stanford are skipped because the stanford parser jar doesn\'t exist')
    
#***********************************************************************
# Test the module
if __name__ == '__main__':
  """"""
  
  import doctest
  doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)