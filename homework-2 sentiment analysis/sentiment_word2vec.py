import matplotlib.pyplot as plt
from sgd import load_saved_params, sgd
from softmaxreg import softmaxRegression, getSentenceFeature, accuracy, softmax_wrapper

from data_utils import *

# Try different regularizations and pick the best!
# NOTE: fill in one more "your code here" below before running!
# Assign a list of floats in the block below
REGULARIZATION = [0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]  
STEP = [1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10] 

# Load the dataset
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# Load the word vectors we trained earlier 
_, wordVectors0, _ = load_saved_params()
wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])
dimVectors = wordVectors.shape[1]

# Load the train set
trainset = dataset.getTrainSentences()
nTrain = len(trainset)
trainFeatures = np.zeros((nTrain, dimVectors))
trainLabels = np.zeros((nTrain,), dtype=np.int32)
for i in range(nTrain):
    words, trainLabels[i] = trainset[i]
    trainFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

# Prepare dev set features
devset = dataset.getDevSentences()
nDev = len(devset)
devFeatures = np.zeros((nDev, dimVectors))
devLabels = np.zeros((nDev,), dtype=np.int32)
for i in range(nDev):
    words, devLabels[i] = devset[i]
    devFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)


################## Try our regularization parameters ##########################
results = []
best_acc = 0
best_weights = None
for regularization in REGULARIZATION:
    random.seed(3141)
    np.random.seed(59265)
    weights = np.random.randn(dimVectors, 5)
    print("Training for reg=%f" % regularization )

    # We will do batch optimization
    weights = sgd(lambda weights: softmax_wrapper(trainFeatures, trainLabels, 
        weights, regularization), weights, 3.0, 15000, PRINT_EVERY=500)

    # Test on train set
    _, _, pred = softmaxRegression(trainFeatures, trainLabels, weights)
    trainAccuracy = accuracy(trainLabels, pred)
    print("Train accuracy (%%): %f" % trainAccuracy)

    # Test on dev set
    _, _, pred = softmaxRegression(devFeatures, devLabels, weights)
    devAccuracy = accuracy(devLabels, pred)
    print("Dev accuracy (%%): %f" % devAccuracy)

    # Save the results and weights
    results.append({
        "reg" : regularization, 
        "weights" : weights, 
        "train" : trainAccuracy, 
        "dev" : devAccuracy})
    
    # the best weights
    if devAccuracy > best_acc:
        best_acc = devAccuracy
        best_reg = regularization
        best_weights = weights

# Print the accuracies
print("")
print("=== Recap ===")
print("Reg\t\tTrain\t\tDev")
for result in results:
    print("%E\t%f\t%f" % (result["reg"],result["train"],result["dev"]))
print()

# Pick the best regularization parameters
BEST_REGULARIZATION = best_reg
BEST_WEIGHTS = best_weights

# Test your findings on the test set
testset = dataset.getTestSentences()
nTest = len(testset)
testFeatures = np.zeros((nTest, dimVectors))
testLabels = np.zeros((nTest,), dtype=np.int32)
for i in range(nTest):
    words, testLabels[i] = testset[i]
    testFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

_, _, pred = softmaxRegression(testFeatures, testLabels, BEST_WEIGHTS)
print("Best regularization value: %E" % BEST_REGULARIZATION)
print("Test accuracy (%%): %f" % accuracy(testLabels, pred))

# Make a plot of regularization vs accuracy
plt.figure(figsize = (15,15))
plt.plot(REGULARIZATION, [x["train"] for x in results])
plt.plot(REGULARIZATION, [x["dev"] for x in results])
plt.xscale('log')
plt.xlabel("regularization")
plt.ylabel("accuracy")
plt.legend(['train', 'dev'], loc='upper left')
plt.savefig("reg_acc.png")
plt.show()


######################## Try our step size parameters #########################
regularization = BEST_REGULARIZATION
results = []
best_acc = 0
best_weights = None
for step in STEP:
    random.seed(3141)
    np.random.seed(59265)
    weights = np.random.randn(dimVectors, 5)
    print("Training for step=%f" % step )

    # We will do batch optimization
    weights = sgd(lambda weights: softmax_wrapper(trainFeatures, trainLabels, 
        weights, regularization), weights, step, 15000, PRINT_EVERY=500)

    # Test on train set
    _, _, pred = softmaxRegression(trainFeatures, trainLabels, weights)
    trainAccuracy = accuracy(trainLabels, pred)
    print("Train accuracy (%%): %f" % trainAccuracy)

    # Test on dev set
    _, _, pred = softmaxRegression(devFeatures, devLabels, weights)
    devAccuracy = accuracy(devLabels, pred)
    print("Dev accuracy (%%): %f" % devAccuracy)

    # Save the results and weights
    results.append({
        "step" : step, 
        "weights" : weights, 
        "train" : trainAccuracy, 
        "dev" : devAccuracy})
    
    # the best weights and step
    if devAccuracy > best_acc:
        best_acc = devAccuracy
        best_step = step
        best_weights = weights

# Print the accuracies
print("")
print("=== Recap ===")
print("Step\t\tTrain\t\tDev")
for result in results:
    print("%E\t%f\t%f" % (result["step"],result["train"],result["dev"]))
print()

# Pick the best step parameters
BEST_STEP = best_step
BEST_WEIGHTS = best_weights

# Test your findings on the test set
testset = dataset.getTestSentences()
nTest = len(testset)
testFeatures = np.zeros((nTest, dimVectors))
testLabels = np.zeros((nTest,), dtype=np.int32)
for i in range(nTest):
    words, testLabels[i] = testset[i]
    testFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

_, _, pred = softmaxRegression(testFeatures, testLabels, BEST_WEIGHTS)
print("Best learning rate: %E" % BEST_STEP)
print("Test accuracy (%%): %f" % accuracy(testLabels, pred))

# Make a plot of step vs accuracy
plt.figure(figsize = (15,15))
plt.plot(STEP, [x["train"] for x in results])
plt.plot(STEP, [x["dev"] for x in results])
plt.xscale('log')
plt.xlabel("learning rate")
plt.ylabel("accuracy")
plt.legend(['train', 'dev'], loc='upper left')
plt.savefig("lr_acc.png")
plt.show()