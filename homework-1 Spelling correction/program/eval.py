import nltk
import os
os.chdir(os.getcwd())

n = 1000

anspath = './ans.txt'
resultpath = './result.txt'
ansfile = open(anspath,'r') # 正确答案
resultfile = open(resultpath,'r') # 模型给出的答案
count=0
for i in range(n):
    # 读入一行并用tab分成句子序号、句子
    ansline = ansfile.readline().split('\t')[1] 
    # 句子分词，并转化为集合
    ansset = set(nltk.word_tokenize(ansline)) 
    
    resultline = resultfile.readline().split('\t')[1]
    resultset = set(nltk.word_tokenize(resultline))
    if ansset == resultset:
        count += 1

    if (i+1)%50 == 0:
        print('第{}个句子评判完毕'.format(i+1))
print("Accuracy is : {:.2f} %".format(count/n *100))