import numpy as np
import random

from gradcheck import gradcheck_naive
from gradcheck import sigmoid ### sigmoid function can be used in the following codes

def softmax(x):
    """
    Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    """
    assert len(x.shape) <= 2

    # let the max one be e^0
    y = np.exp(x - np.max(x, axis=len(x.shape) - 1, keepdims=True))
    # the sum of all e^(y)
    normalization = np.sum(y, axis=len(x.shape) - 1, keepdims=True)
    return np.divide(y, normalization)


def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    
    ### YOUR CODE HERE
    norm = np.linalg.norm(x, axis = 1)
    x = (x.T/norm).T - 1e-15
    ### END YOUR CODE
    
    return x

def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print(x)
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print()

def softmaxCostAndGradient(predicted, target, outputVectors, dataset = None):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector                                                
    # - grad: the gradient with respect to all the other word        
    #        vectors
    
    ### YOUR CODE HERE
    
    # probabilities of words, Mx1 = 1xN dot NxM
    p = softmax(predicted.dot(outputVectors.T))
    cost = -np.log(p[target])
    
    M = p.shape[0] # number of words
    N = predicted.shape[0] # dim of a word vector

    p[target] -= 1 # for u0 term, it have more -Vc (predicted) than other u's
    
    # turn 1xN matrix into a vector
    gradPred = (p.reshape((1,M)).dot(outputVectors)).flatten()
    
    # MxN matrix
    grad = p.reshape((M,1)) * predicted.reshape((1,N))

    ### END YOUR CODE

    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient
    
    ### YOUR CODE HERE
    
    # initiate gradient for other (neg) samples and predicted sample
    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)
    
    # calculate useful sigmoid value and 1-sigmoid value
    s = sigmoid(predicted.dot(outputVectors[target,:]))
    t = 1 - s
    
    # initiate cost function and renew the 1st step (pos sample) for grads
    cost = -np.log(s) # cost = -yilog(s)
    gradPred -= t * outputVectors[target, :]
    grad[target, :] -= t * predicted
 
    # sample K times for neg samples
    for k in range(K):
        neg = dataset.sampleTokenIdx()
        
        # here 1 - Sigmoid(x) = Sigmoid(-x)
        s = sigmoid(-predicted.dot(outputVectors[neg,:]))
        t = 1 - s
        cost += -np.log(s) # cost = -sum((1-yi)log(s)), yi=0
 
        gradPred += t * outputVectors[neg, :]
        # only neg grads will renew
        grad[neg, :] += t * predicted
    ### END YOUR CODE
    
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors

    ### YOUR CODE HERE
    current_index = tokens[currentWord]
    predicted = inputVectors[current_index, :]
 
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    
    # for every word, calculate its cost and grad w.r.t. word vectors
    for word in contextWords:
        
        # get the index of the word
        index = tokens[word]
        
        # calculate cost and grads 
        c, gi, go = word2vecCostAndGradient(predicted, index, outputVectors, dataset)
        
        # add cost and grad of u to cost and gradOut
        cost += c
        gradOut += go
        
        # for predicted word, calculate its cumulative gradient with words in context
        gradIn[current_index, :] += gi
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N//2,:]
    outputVectors = wordVectors[N//2:,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N//2, :] += gin / batchsize / denom
        grad[N//2:, :] += gout / batchsize / denom
        ### we use // to let N/2  be int
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    #print("\n==== Gradient check for CBOW      ====")
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))
    #print(cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    #print(cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()