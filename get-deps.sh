#!/bin/bash

# Requires package "openjdk-7-jdk"

# This is probably subject to change--fill it in with info from
# http://nlp.stanford.edu/software/lex-parser.shtml
DATE=2015-04-20
VERSION=3.5.2

wget "http://nlp.stanford.edu/software/stanford-parser-full-$DATE.zip"
unzip "stanford-parser-full-$DATE.zip"
cd stanford-parser-full-$DATE
jar xf stanford-parser-$VERSION-models.jar