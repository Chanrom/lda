#!/usr/bin/python
#coding=utf-8

import logging
from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = ["Human machine interface for lab abc computer applications", \
"A survey of user opinion of computer system response time",\
"The EPS user interface management system",\
"System and human system engineering testing of EPS",\
"Relation of user perceived response time to error measurement",\
"The generation of random binary unordered trees",\
"The intersection graph of paths in trees",\
"Graph minors IV Widths of trees and well quasi ordering",\
"Graph minors A survey"]

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())

texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

# remove words that appear only once
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once] for text in texts]

print 'texts:\n', texts

dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester.dict') # store the dictionary, for future reference

print 'dictionary:\n', dictionary

print 'dictionary token2id:\n', dictionary.token2id

new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print 'new_vec:\n', new_vec # the word "interaction" does not appear in the dictionary and is ignored

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus) # store to disk, for later use
print 'corpus:\n', corpus
