#!/usr/bin/env python

import pickle as pkl
import re
import os, os.path

#mypath = 'dicts/en'
#files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
#
#for filename in files:
#  temp = set()
#  with open(os.path.join(mypath, filename)) as f:
#    for line in f:
#      temp.add(line.strip())
#  pkl.dump(temp, open(os.path.join(mypath, filename[:-3]+'pkl'), 'w'))

regex0 = re.compile('(.*?)  (.*?)  /# (.*?) #/  (.*?)')
regex1 = re.compile('(\[.*?\])')
temp = set()
seen = set()
s = ''
with open('dicts/en/sylcmu.txt') as f:
  for line in f:
    line = regex0.match(line)
    if line:
      ortho  = line.group(1)
      segs   = line.group(2)
      sylls  = line.group(3)
      weight = line.group(4)
      if ortho not in seen:
        seen.add(ortho)
        if len(regex1.findall(sylls)) == 1:
          temp.add(ortho)
          s = s+ortho+'\n'
  pkl.dump(temp, open('dicts/en/monosyl.pkl', 'w'))
with open('dicts/en/monosyl.txt', 'w') as f:
  f.write(s)
