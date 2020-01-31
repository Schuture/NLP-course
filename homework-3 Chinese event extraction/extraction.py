'''
Extraction of trigger words and event words with two methods:
The first one is Hidden Markov Model in detailed python codes
The second one is CRF model in package
'''

import time
import numpy as np
from nltk import ngrams
from collections import defaultdict

# hyperparameters
a1 = 0.6
a2 = 0.4


def DataLoader(train_address, test_address):
    '''
    Load one dataset including training set and testing set, process them
    
    Input:
        train_address: the address of training set
        test_address: the address of testing set
    Output:
        transition: the transition probabilities, p(v,s)
        emission: the emission probabilities, e(x|s)
        prior: the prior probabilities of tags, pi(s)
        tags: tuple containing all tags
        test_sent: testing sentences in a list
        test_tags: testing tags in a list
    '''
    ########### first part, load training data and get probabilities ##########
    transition = defaultdict(int)
    emission = defaultdict(int)
    prior = defaultdict(int)
    # counters for c(v,s), c(x,s), c(s)
    cvs = defaultdict(int)
    cxs = defaultdict(int)
    cs = defaultdict(int)
    words = set() # bag of words
    
    sentence = []
    tags = []
    
    def count(sentence, tags):
        bigrams = ngrams(tags, 2)
        for bigram in bigrams:
            cvs[bigram] += 1
        # c(x,s)
        n = len(sentence)
        for i in range(n):
            cxs[(sentence[i], tags[i])] += 1
            cs[tags[i]] += 1
        
    # read training file / train model
    with open(train_address, encoding='utf-8', errors='ignore') as train_f:
        while True:
            datum = train_f.readline()
            if not datum:
                count(sentence, tags)
                break
            if datum == '\n':
                count(sentence, tags)
                sentence = []
                tags = []
            else:
                word, tag = datum.split('\t')
                tag = tag[:-1] # remove \n
                words.add(word)
                sentence.append(word)
                tags.append(tag)
                
    tags = tuple(cs.keys()) # we use it as a tag container previously
    words = tuple(words) # all different words
    
    # calculating probabilities
    total = sum(cs.values())
    for key in cs: # prior probabilities
        prior[key] = cs[key] / total
    
    for tag1 in tags: # transition probabilities, tag2 -> tag1
        for tag2 in tags:
            # a1 * p(tag1|tag2) + a2 * p(tag1), back-off smoothing
            transition[(tag1, tag2)] = a1 * cvs[(tag1, tag2)] / cs[tag2] + a2 * prior[tag1]
        
    for word in words: # emission probabilities, add-1 smoothing
        for tag in tags:
            emission[(word, tag)] = cxs[(word, tag)] / cs[tag]
        
    ######################### second part, load test data #####################
    test_sent = []
    test_tags = []
    sentence = []
    tags = []
    # read testing file
    with open(test_address, encoding='utf-8', errors='ignore') as train_f:
        while True:
            datum = train_f.readline()
            if not datum:
                test_sent.append(sentence)
                test_tags.append(tags)
                break
            if datum == '\n':
                test_sent.append(sentence)
                test_tags.append(tags)
                sentence = []
                tags = []
            else:
                word, tag = datum.split('\t')
                tag = tag[:-1] # remove \n
                sentence.append(word)
                tags.append(tag)
                
    tags = tuple(cs.keys()) # we use it as a tag container previously
    
    return transition, emission, prior, tags, test_sent, test_tags
    

def Viterbi(sentences, transition, emission, prior, tags):
    '''
    Viterbi algorithm for generate tags for sentences
    
    Input:
        sentences: sentences for tagging, List[List[]]
        transition: transition probabilities for tags
        emission: emission probabilities for tags
        tags: tag categories, a set
    Output:
        result_tags: tags for input sentences, List[List[]]
    '''
    tags = tuple(tags)
    num_tags = len(tags)
    result_tags = []
    
    def find_index(col): # find the best tag for this word
        index = 0
        prob = - 3.14e100
        for j in range(num_tags):
            if dp[col][j] > prob:
                prob = dp[col][j]
                index = j
        return index
    
    for sentence in sentences:
        n = len(sentence)
        if n == 0:
            continue
        dp = np.zeros((n, num_tags))
        # initialization, with log probability
        for j in range(num_tags):
            if emission[(sentence[0], tags[j])] == 0: 
                emission[(sentence[0], tags[j])] = 3.14e-100 # the smallest fp
            dp[0][j] = np.log(prior[tags[j]] * emission[(sentence[0], tags[j])])
        # fill the table, with log probability
        for i in range(1, n):
            for j in range(num_tags):
                # very few samples are in the testset but not in the training set
                if emission[(sentence[i], tags[j])] == 0: 
                    emission[(sentence[i], tags[j])] = 3.14e-100
                dp[i][j] = max([dp[i-1][k] + np.log(transition[(tags[j],tags[k])]) + \
                               np.log(emission[(sentence[i], tags[j])]) for k in range(num_tags)])
        # now we have a complete table of probabilities, backtrack
        index_series = [find_index(col) for col in range(n)]
        tag_series = [tags[idx] for idx in index_series]
        result_tags.append(tag_series)
        
    return result_tags


def ResultSaver(test_sent, test_tags, result_tags, result_address):
    '''
    Open a result file and save the predicted tags along with the correct data
    '''
    print('=================== start saving results ==================')
    with open(result_address,'w', encoding='utf-8') as result_file:
        for i in range(len(result_tags)):
            for j in range(len(result_tags[i])):
                result_file.write('{}\t{}\t{}\n'.format(test_sent[i][j], test_tags[i][j], result_tags[i][j]))
            result_file.write('\n')


def test(train_address, test_address, result_address):
    '''
    run the algorithm with one of the two datasets, and save the result
    '''
    transition, emission, prior, tags, test_sent, test_tags = DataLoader(train_address, test_address)
    print('Loading and training complete')
    result_tags = Viterbi(test_sent, transition, emission, prior, tags)
    print('Viterbi decoding complete')
    ResultSaver(test_sent, test_tags, result_tags, result_address)
    print('Complete saving results!')
            
    
if __name__ == '__main__':
    start = time.time()
    train_address = 'trigger_train.txt'
    test_address = 'trigger_test.txt'
    result_address = 'trigger_result.txt'
    print('=============== training trigger classifier ===============')
    test(train_address, test_address, result_address)
    print('Consume: {} seconds'.format(round(time.time() - start,4)))
    
    start = time.time()
    train_address = 'argument_train.txt'
    test_address = 'argument_test.txt'
    result_address = 'argument_result.txt'
    print('\n=============== training argument classifier ==============')
    test(train_address, test_address, result_address)
    print('Consume: {} seconds'.format(round(time.time() - start,4)))















