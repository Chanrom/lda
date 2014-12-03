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

def getCorpusText(train_file, train):
    train_lines = codecs.open(train_file, 'r', encoding='utf-8').readlines()    
    for index in range(1, len(train_lines)):
        try:
             ##get trainning text
            line_split = train_lines[index].split('\t')
            short_text1 = line_split[3].strip('r\n').strip('\n')
            short_text2 = line_split[4].strip('\r\n').strip('\n')
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
    token_document_count = {} ##key=token, value=the number of documents the token occurs
    for item in train:
        text_list1 = cleanText(item[0])
        text_list2 = cleanText(item[1])

        ## count tokens occur
        for token in set(text_list1):
            if token_document_count.has_key(token):
                token_document_count[token] += 1
            else:
                token_document_count[token] = 1
        for token in set(text_list2):
            if token_document_count.has_key(token):
                token_document_count[token] += 1
            else:
                token_document_count[token] = 1

        if flag:
            train_bag.append((text_list1, text_list2))
        else:
            train_bag.append(text_list1)
            train_bag.append(text_list2)
        
    return token_document_count

def getLDATopic(train_corpus, token_document_count):
    train_corpus_bag = []
    token_document_count = getWordBagModel(train_corpus, train_corpus_bag, False) ##word bag model
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

def convertList2Hash(l):
    h = {}
    for item in l:
        h[item[0]] = item[1]
    
    return h

def calculateTfidf(text_vec, token_list, text_dic, train_text_number, token_text_count):
    for item in token_list:
        tf_idf = 0
        token = item[0]
        token_index = item[1]
        if text_dic.has_key(token_index) and token_text_count.has_key(token):
            print 'come on'
            tf = text_dic[token_index]
            idf = math.log(train_text_number / token_text_count[token], 2)
            tf_idf = tf * idf

        text_vec.append(tf_idf)

def calculateSimi(test_corpus, lda, train_text_number, token_text_count):
    test_corpus_bag = []
    getWordBagModel(test_corpus, test_corpus_bag, False)
    
    for index in range(0, len(test_corpus_bag), 2):
        text_list1 = test_corpus_bag[index]
        text_list2 = test_corpus_bag[index + 1]
        
        dictionary = corpora.Dictionary([text_list1, text_list2])

        sorted_dict = sorted(dictionary.token2id.iteritems(), key = lambda d:d[1], reverse = False)
        print sorted_dict

        text_vec_hash1 = convertList2Hash(dictionary.doc2bow(text_list1))
        text_vec_hash2 = convertList2Hash(dictionary.doc2bow(text_list2))
        
        print text_vec_hash1
        print text_vec_hash2
        
        text_vec1 = []
        text_vec2 = []
        ## calculate tf-idf
        calculateTfidf(text_vec1, sorted_dict, text_vec_hash1, train_text_number, token_text_count)
        calculateTfidf(text_vec2, sorted_dict, text_vec_hash2, train_text_number, token_text_count)

        print 'text_vec1:', text_vec1
        print 'text_vec2:', text_vec2


#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

train_corpus = []
getCorpusText(train_file, train_corpus) ##get train corpus

token_document_count = {}
lda = getLDATopic(train_corpus, token_document_count) ##lda model for getting train corpus's topics

lda.save('../data/lda.model')

print 'topic 1, 10 words:', lda.show_topic(1, 10)

test_corpus = []

getCorpusText(test_file, test_corpus) ##get test corpus

calculateSimi(test_corpus, lda, 2 * len(train_corpus), token_document_count) ##calculate similarity of sentence pairs


