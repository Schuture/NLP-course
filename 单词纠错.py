import collections
from nltk.corpus import brown
import os
os.chdir('homework-1 Spelling correction/program') # 工作目录

dic_path = './vocab.txt' # 词典路径
dic = set() # 空集合，用来存放词典
with open(dic_path) as dictionary:
    while True:
        word = dictionary.readline()
        if not word:
            break
        if word.endswith('\n'):
            dic.add(word[:-1]) # 将单词加入词典
        else:
            dic.add(word)

training_data = brown.words(categories = 'news')
training_data = [word.lower() for word in training_data]

# get P(w)，即每个词出现的频次
nwords = collections.Counter(training_data)

alpha = 'abcdefghijklmnopqrstuvwxyz'
#一步调整，编辑距离为1
def edit1(word): 
    n = len(word)
    add_a_char = [word[0:i] + c + word[i:] for i in range(n+1) for c in alpha]
    delete_a_char = [word[0:i] + word[i+1:] for i in range(n)]
    revise_a_char = [word[0:i] + c + word[i+1:] for i in range(n) for c in alpha]
    swap_adjacent_two_chars = [word[0:i] + word[i+1]+ word[i]+ word[i+2:] for i in range(n-1)] 
    return set( add_a_char + delete_a_char +
               revise_a_char +  swap_adjacent_two_chars)

# 两步调整，编辑距离为2
def edit2(word):
    return set(e2 for e1 in edit1(word) for e2 in edit1(e1))


# 朴素贝叶斯分类器

def identify(words): # 取出words中的正确词
    return set(w for w in words if w in dic)

def getMax(wanteds): # 从编辑后的词中选出
    threewanteds=[]
    maxword = max(wanteds,key=lambda w : nwords[w]) # 备选词中选出P(w)最大的
    threewanteds.append('want to input: '+ maxword)
    wanteds.remove(maxword)
    if len(wanteds)>0:
        maxword = max(wanteds,key=lambda w : nwords[w])
        threewanteds.append(maxword)
        wanteds.remove(maxword)
        if len(wanteds)>0:
            maxword = max(wanteds,key=lambda w : nwords[w])
            threewanteds.append(maxword)   
    return threewanteds

def bayesClassifier(word):
    #如果字典中有输入的单词，直接返回
    if identify([word]):
        return 'found: '+ word
    #一步调整
    wanteds = identify(edit1(word)) 
    if len(wanteds)>0:
        return getMax(wanteds)
    #两步调整
    wanteds = identify(edit2(word))
    if len(wanteds)>0:
        return getMax(wanteds)
    #不再修正，直接提示这个单词不在当前的词典中
    else:    
        return [word + ' not found in dictionary!' ]
    
print(bayesClassifier('pigg'))
print(bayesClassifier('clasroom'))
print(bayesClassifier('stdio'))
print(bayesClassifier('hypotheses'))
print(bayesClassifier('bvjfkapbaij'))
