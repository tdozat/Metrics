#!/usr/bin/env python

import pickle as pkl
import os, os.path

mypath = 'dicts/en'
files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

for filename in files:
  temp = set()
  with open(os.path.join(mypath, filename)) as f:
    for line in f:
      temp.add(line.strip())
  pkl.dump(temp, open(os.path.join(mypath, filename[:-3]+'pkl'), 'w'))

