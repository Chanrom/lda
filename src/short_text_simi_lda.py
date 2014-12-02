#!/usr/bin/python
#coding=utf-8

import sys
import codecs
import logging
from gensim import corpora, models, similarities
from nltk import corpus

try:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
except:
    print 'Usage: python short_text_simi_lda.py train_file test_file\n'
    exit()

def getTrainText(train_file, train):
    train_lines = codecs.open(train_file, 'r', encoding='utf-8').readlines()    
    for index in range(1, len(train_lines)):
        try:
             ##get trainning text
            line_split = train_lines[index].split('\t')
            short_text1 = line_split[3]
            short_text2 = line_split[4]
        except Exception, e:
            print e
            continue
        else:
            train.append((short_text1, short_text2))

def cleanText(text): ## remove common words and tokenize
    word_list = text.lower().replace('.', '').replace(',', '')\
.replace('?', '').replace(':', '').replace('!', '').replace('-', '')\
.replace('(', '').replace(')', '').replace('  ', ' ').split(' ')
    
    return [word for word in word_list if not word in corpus.stopwords.words('english')]

def getWordBagModel(train, train_bag, flag):  ##if flag == False, that means no neating list; else has neating list.
    for item in train:
        text_list1 = cleanText(item[0])
        text_list2 = cleanText(item[1])
        if flag:
            train_bag.append((text_list1, text_list2))
        else:
            train_bag.append(text_list1)
            train_bag.append(text_list2)

def getLDATopic(train_corpus_bag):
    dictionary = corpora.Dictionary(train_corpus_bag)
    dictionary.save('../data/train_corpus.dict') # store the dictionary, for future reference

    ##print dictionary
    ##print dictionary.token2id
    ##new_doc = "Human computer interaction"
    ##new_vec = dictionary.doc2bow(new_doc.lower().split())
    ##print new_vec # the word "interaction" does not appear in the dictionary and is ignored\

    corpus = [dictionary.doc2bow(text) for text in train_corpus_bag]

    corpora.MmCorpus.serialize('../data/train_corpus.mm', corpus) # store to disk, for later use

    mm = corpora.MmCorpus('../data/train_corpus.mm')
    lda = models.ldamodel.LdaModel(corpus = mm, id2word = dictionary, num_topics = 100)

    #lda.print_topics(100)
    
    return lda
    

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

train_corpus = []

getTrainText(train_file, train_corpus) ##get train corpus

train_corpus_bag = []
getWordBagModel(train_corpus, train_corpus_bag, False) ##word bag model

lda = getLDATopic(train_corpus_bag) ##lda model for getting train corpus's topics

lda.save('../data/lda.model')

print 'topic 1, 10 words:', lda.show_topic(1, 10)
