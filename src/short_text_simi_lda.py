#!/usr/bin/python
#coding=utf-8

import sys
import codecs
import logging
import math
from gensim import corpora, models, similarities
from nltk import corpus
from scipy import spatial

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

def getLDATopic(train_corpus):

    global NUM_TOPICS

    train_corpus_bag = []
    token_document_count = getWordBagModel(train_corpus, train_corpus_bag, False) ##word bag model

    dictionary = corpora.Dictionary(train_corpus_bag)
    dictionary.save('../data/corpus.dict') # store the dictionary, for future reference

    corpus = [dictionary.doc2bow(text) for text in train_corpus_bag]

    corpora.MmCorpus.serialize('../data/corpus.mm', corpus) # store to disk, for later use

    mm = corpora.MmCorpus('../data/corpus.mm')
    lda = models.ldamodel.LdaModel(corpus = mm, id2word = dictionary, num_topics = NUM_TOPICS)

    #lda.print_topics(100)
    return (lda, token_document_count)

def convertList2Hash(l):
    h = {}
    for item in l:
        h[item[0]] = item[1]
    
    return h

def calculateTfidf(text_vec, token_list, text_dic, train_text_number, token_text_count):
#    print 'token_list:', token_list
#    print 'text_vec:', text_vec
#    print 'text_dic:', text_dic
#    print 'train_text_number:', train_text_number
#    print 'token_text_count:', token_text_count
    for item in token_list:
        tf_idf = 0
        token = item[0]
        token_index = item[1]
        if text_dic.has_key(token_index) and token_text_count.has_key(token):
            tf = text_dic[token_index]
            idf = math.log(train_text_number / token_text_count[token], 2)
            tf_idf = tf * idf

        text_vec.append(tf_idf)

def getTopics(lda, topics):
    
    global NUM_TOPICS

    for i in range(NUM_TOPICS):
        words = {}
        topic = lda.show_topic(i, 1000)
        for item in topic:
            if item[0] >= 0.001:
                words[item[1]] = item[0]
        topics.append(words)

def getProAndIndex(text_dist, word_list, topic):
    max_pro_token = ''
    max_token_index = -1
    max_pro = 0
    for index in text_dist:
        token = word_list[index][0]
        pro = topic[token] if topic.has_key(token) else 0
        if pro > max_pro:
            max_pro_token = token
            max_token_index = index
            max_pro = pro
#    print max_pro, max_token_index
    return (max_pro, max_token_index)

def getMaxProAndIndex(word_list, text_hash1, text_hash2, topic):
#    print word_list
#    print text_hash1
#    print text_hash2
#    print topic
    text_dist1 = []
    text_dist2 = []
    for item in word_list:
        token = item[0]
        index = item[1]
        if text_hash1.has_key(index) and (not text_hash2.has_key(index)):
            text_dist1.append(index)
        elif text_hash2.has_key(index) and (not text_hash1.has_key(index)):
            text_dist2.append(index)
    
    (max_pro1, max_token_index1) = getProAndIndex(text_dist1, word_list, topic)
    (max_pro2, max_token_index2) = getProAndIndex(text_dist2, word_list, topic)
    
    return [(max_pro1, max_token_index1), (max_pro2, max_token_index2)]


def modifyCosineSimi(mass_data, lda, test_corpus_simis):
 
    topics = []
    getTopics(lda, topics)

    for topic in topics:
        for text_data in mass_data:
            
            
            text_vec1 = text_data[0]
            text_vec2 = text_data[1]
            word_list =  text_data[2]
            text_hash1 = text_data[3]
            text_hash2 = text_data[4]
            
            pro_index_data = getMaxProAndIndex(word_list, text_hash1, text_hash2, topic)
            max_pro1 = pro_index_data[0][0]
            max_token_index1 = pro_index_data[0][1]
            max_pro2 = pro_index_data[1][0]
            max_token_index2 =  pro_index_data[1][1]
            if max_token_index1 != -1 and max_token_index2 != -1 and max_pro1 > 0.05 and max_pro2 > 0.05:
                print 'modify'
                print topic
                print word_list
                print text_vec1
                print text_vec2
                print max_token_index1, max_pro1
                print max_token_index2, max_pro2
                print text_vec1[max_token_index2], text_vec2[max_token_index2]
                print text_vec2[max_token_index1], text_vec1[max_token_index1]
                text_vec1[max_token_index2] += text_vec2[max_token_index2] * max_pro1
                text_vec2[max_token_index1] += text_vec1[max_token_index1] * max_pro2
                print text_vec1
                print text_vec2
                print text_vec1[max_token_index2]
                print text_vec2[max_token_index1]
            
    for text_data in mass_data:
        text_vec1 = text_data[0]
        text_vec2 = text_data[1]
        test_corpus_simis.append(1 - spatial.distance.cosine(text_vec1, text_vec2))

def calculateSimi(test_corpus, lda, train_text_number, token_text_count):

    test_corpus_simis = []  ##similarity for each pair of sentences
    mass_data = [] 

    test_corpus_bag = []
    getWordBagModel(test_corpus, test_corpus_bag, False)
    
    for index in range(0, len(test_corpus_bag), 2):
        text_list1 = test_corpus_bag[index]
        text_list2 = test_corpus_bag[index + 1]
        
        dictionary = corpora.Dictionary([text_list1, text_list2])

        sorted_dict = sorted(dictionary.token2id.iteritems(), key = lambda d:d[1], reverse = False)
#        print 'sorted_dict:', sorted_dict

        text_vec_hash1 = convertList2Hash(dictionary.doc2bow(text_list1))
        text_vec_hash2 = convertList2Hash(dictionary.doc2bow(text_list2))
        
#        print 'text_vec_hash1:', text_vec_hash1
#        print 'text_vec_hash2:', text_vec_hash2

        ## VSM Model, calculate tf-idf        
        text_vec1 = []
        text_vec2 = []
        calculateTfidf(text_vec1, sorted_dict, text_vec_hash1, train_text_number, token_text_count)
        calculateTfidf(text_vec2, sorted_dict, text_vec_hash2, train_text_number, token_text_count)

        mass_data.append((text_vec1, text_vec2, sorted_dict, text_vec_hash1, text_vec_hash2))

    modifyCosineSimi(mass_data, lda, test_corpus_simis)

    return test_corpus_simis


#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

NUM_TOPICS = 100

train_corpus = []
getCorpusText(train_file, train_corpus) ##get train corpus
test_corpus = []
getCorpusText(test_file, test_corpus) ##get test corpus

(lda, token_document_count) = getLDATopic(train_corpus + test_corpus) ##lda model for getting train corpus's topics
#print 'token_document_count:', token_document_count
lda.save('../data/lda.model')

print 'topic 1, 10 words:', lda.show_topic(1, 10)


test_corpus_simi =  calculateSimi(test_corpus, lda, 2 * (len(train_corpus) + len(test_corpus)), token_document_count) ##calculate similarity of sentence pairs

test_result = []
for simi in test_corpus_simi:
    if simi >= 0.5:
        test_result.append(1)
    else:
        test_result.append(0)

gs_lines = codecs.open('../data/gold.result', 'r', encoding='utf-8').readlines()
sum_gs = 0
tt = 0
for index in range(len(gs_lines)):
    if test_result[index] == 1 and int(gs_lines[index][0]) == 1:
        tt += 1
        sum_gs += 1
    elif int(gs_lines[index][0]) == 1:
        sum_gs += 1
print tt
print sum(test_result)
print sum_gs

P = float(tt) / sum(test_result)
R = float(tt) / sum_gs
print '\nP:', P
print 'R:', R
print 'F', 2 * P * R / (P + R)


f = codecs.open('../data/test.result', 'w', encoding='utf-8')
f.write('\n'.join([str(fl) for fl in test_corpus_simi]))
f.close()


