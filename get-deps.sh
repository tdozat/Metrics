#!/bin/bash


# This is probably subject to change--fill it in with info from
# http://nlp.stanford.edu/software/lex-parser.shtml
DATE=2015-04-20
VERSION=3.5.2

wget "http://nlp.stanford.edu/software/stanford-parser-full-$DATE.zip"
unzip "stanford-parser-full-$DATE.zip"
cd stanford-parser-full-$DATE
jar xf stanford-parser-$VERSION-models.jar

su
add-apt-repository ppa:webupd8team/java
apt-get update
apt-get install oracle-java8-installer
exit