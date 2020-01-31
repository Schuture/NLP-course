import nltk
from nltk.corpus import names
import random

def gender_features(word):
    feature = dict()
    feature['last_letter'] = word[-1]
    feature['last_2letter'] = word[-2:]
    return feature

male_names = [(name, 'male') for name in names.words('male.txt')]
female_names = [(name, 'female') for name in names.words('female.txt')]
labeled_names = male_names + female_names
random.shuffle(labeled_names)

# 提取特征、标签
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

# 分割训练集、测试集
train_set, test_set = featuresets[:int(0.7*len(labeled_names))], \
                    featuresets[int(0.7*len(labeled_names)):]
                    
# 训练朴素贝叶斯分类器
classifier = nltk.NaiveBayesClassifier.train(train_set)

# 测试"Sam"的预测结果
print(classifier.classify(gender_features('Sam')))

# 测试分类器，得到准确率
print(nltk.classify.accuracy(classifier, test_set))

# 效用最高的五个特征
print(classifier.show_most_informative_features(5))