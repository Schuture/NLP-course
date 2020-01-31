from collections import Counter
from data_utils import *
from softmaxreg import accuracy

def naiveBayesFeature(tokens, sentence):
    '''
    extract feature based on bag of word
    
    Input:
        tokens: a dictionary that maps words to their indices in    
                  the word vector list 
        sentence: a list of words in the sentence of interest 
    Output:
        feature: a dict, word frequency of a sentence
    '''
    # get all indices of words in a sentence
    indices = [tokens[word] for word in sentence]
    features = dict(Counter(indices))
    
    return features


def naiveBayesClassifier(tokens, sentence, prior, conditional):
    '''
    naive Bayes classifier for sentiment analysis
    
    Input:
        sentence: a list of words in the sentence of interest 
        prior: prior possibility of all classes
        conditional: conditional possibility of words of all classes
    Output:
        Class: the classification result
    '''
    features = naiveBayesFeature(tokens, sentence)
    logp = np.zeros(5) # log possibility of each class
    for i in range(5):
        for feature in features.keys():
            # the log conditional p of this word * the number of this word
            logp[i] += np.log(conditional[i][feature]) * features[feature]
        logp[i] += prior[i]
    
    return np.argmax(logp)
        

def train_classifier():
    '''
    train a naive Bayes classifier
    
    Input: None
    Output:
        labels: the frequency of each class, prior possibility
        word_freq: the word frequency of each class, conditional possibility
    '''
    tokens = dataset.tokens()
    nWords = len(tokens) # number of different tokens
    
    trainset = dataset.getTrainSentences() # sentences for training
    nTrain = len(trainset)
    
    labels = [0] * 5 # count labels for calculating prior possibility
    words_class = [0] * 5 # count words in each class
    word_freq = [dict(),dict(),dict(),dict(),dict()] # conditional possibility
    
    # add 1 smoothing and initialization
    for word in tokens.keys():
        index = tokens[word]
        for i in range(5):
            word_freq[i][index] = 1
    
    # count word number for each class
    for i in range(nTrain):
        words, label = trainset[i]
        labels[label] += 1
        for word in words:
            words_class[label] += 1
            index = tokens[word]
            word_freq[label][index] += 1
    
    # transfer number to frequency
    for i in range(5):
        labels[i] /= nTrain # the proportion of each class
        words_this_class = words_class[i]
        denorm = words_this_class + nWords
        for index in word_freq[i].keys():
            word_freq[i][index] /= denorm
            
    return labels, word_freq


def test_classifier(dataset, model):
    ''' 
    test the trained naive Bayes classifier
    
    Input:
        data: testing data
        model: a tuple, trained prior and conditional possibilities
    Output:
        None, but print the accuracy of the model on testing data
    '''
    prior, conditional = model
    
    tokens = dataset.tokens()
    testset = dataset.getTestSentences()
    nTest = len(testset)
    
    pred = np.zeros((nTest,), dtype=np.int32) # prediction
    testLabels = np.zeros((nTest,), dtype=np.int32) # true label
    for i in range(nTest):
        words, testLabels[i] = testset[i]
        pred[i] = naiveBayesClassifier(tokens, words, prior, conditional)
        
    print("Test accuracy (%%): %f" % accuracy(testLabels, pred))
    
                
                
if __name__ == '__main__':
    dataset = StanfordSentiment()
    
    import time
    start = time.time()
    print('=== Training start === \n')
    model = train_classifier()
    print('Training complete!')
    
    print('=== Testing start === \n')
    test_classifier(dataset, model)
    print('Time consumption: {}'.format(time.time() - start))
    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                