#!/usr/bin/env python
import re
import pickle as pkl
import sys

lastWord = 'octopus'
data = {}
info_regex = re.compile(r'(\S+)\s+((?:(?:\S+) ??)+)\s+/# ((?:(?:(?:\[ (?:(?:\S+) ??)+? \]) ??)|(?:\S+) ??)+?) #/\s+S:(\w*) W:(\w*)')
phoneme_regex = re.compile(r'(\S+)')
syllable_regex = re.compile(r'(?:\[ ((?:(?:\S+) ?)+?) ?\])+')
i = 0
with open('sylcmu.txt') as f:
  for line in f:
    if not (line.startswith('##') or line.startswith(lastWord)):
      datum     = info_regex.match(line.strip())
      word      = datum.group(1)
      phonemes  = phoneme_regex.findall(datum.group(2))
      syllables = syllable_regex.findall(datum.group(3))
      syllab    = [phoneme_regex.findall(syllable) for syllable in syllables]
      stress    = list(datum.group(4))
      weight    = list(datum.group(5))
      datum_info = [phonemes, zip(syllab, stress, weight)]
      data[word.lower()] = datum_info
      lastWord = word+' '
      i+=1
      print '%d: %s        \r' % (i, word),
      sys.stdout.flush()
pkl.dump(data, open('sylcmu.pkl', 'w'))