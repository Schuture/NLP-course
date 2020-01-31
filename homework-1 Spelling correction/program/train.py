'''用language model和channel model实现拼写纠错'''
import nltk
import collections
from nltk.corpus import brown
from nltk.corpus import reuters
from nltk.util import ngrams
import os
os.chdir(os.getcwd()) # 工作目录

#################################### 超参数 ###################################
m = 10 # 找句子中真词的错误时，最多用几个词计算confidence
n = 25 # 已知错词位置时，用几个备选来计算language model概率

################################### 读取数据 ###################################

dic_path = './vocab.txt' # 词典路径
test_data_path = './testdata.txt' # 测试文档路径
count_1edit = './count_1edit.txt' # 1 edit distance errors

dic = set() # 空集合，用来存放词典
with open(dic_path) as dictionary:
    while True:
        word = dictionary.readline()
        if not word:
            break
        if word.endswith('\n'):
            dic.add(word[:-1].lower()) # 将单词加入词典,全部改为小写
        else:
            dic.add(word)
        
mistakes = [] # 存放句子的错误数量
sentences = [] # 存放句子
with open(test_data_path) as data_to_read:
    while True:
        line = data_to_read.readline()
        if not line:
            break
        data_line = line.split('\t') # tab为分隔符，分隔出行号，错误数，句子
        mistakes.append(int(data_line[1]))
        if data_line[2].endswith('\n'):
            sentences.append(data_line[2][:-1]) # 去掉结尾换行符，它只算一个字符
        else:
            sentences.append(data_line[2])
            
# 编辑距离为1的混淆频数
errors = [] # [错误字符，正确字符]
error_nums = [] # 对应错误出现的次数
error_num = 0 # 总错误数
with open(count_1edit) as count_1:
    while True:
        line = count_1.readline()
        if not line:
            break
        error_line = line.split('\t') # tab seperation
        error = error_line[0] # type of error
        num = int(error_line[1]) # error number
        
        error_char = error.split("|")[0].lower()    # 为了简单起见，全部转化为小写
        if error_char == ' ':
            error_char = '' # 文件中空格代表缺失，而不是真的空格
        correct_char = error.split("|")[1].lower()  # 为了简单起见，全部转化为小写
        if correct_char == ' ':
            correct_char = '' # 文件中空格代表缺失，而不是真的空格
            
        errors.append([error_char, correct_char])
        error_nums.append(num)
        error_num += num

print('数据读取完毕\n\n')
################################### 定义改词方法 ###########################################

alpha = 'abcdefghijklmnopqrstuvwxyz- ' # 包括空格，连字符

############# 这里的两个大写开头的 Edit, GetMax供纠正真实词使用

def Edit(word):  # 只编辑一个地方，返回格式[[修改后的词，改后字母，改前字母],[],...]
    n = len(word)
    add_a_char = [[word[0:i] + c + word[i:], '{}'.format(c), ''] for i in range(n+1) for c in alpha]
    delete_a_char = [[word[0:i] + word[i+1:], '', word[i]] for i in range(n)]
    revise_a_char = [[word[0:i] + c + word[i+1:], '{}'.format(c), word[i]] for i in range(n) for c in alpha]
    swap_adjacent_two_chars = [[word[0:i] + word[i+1] + word[i] + word[i+2:], word[i+1] + word[i], word[i:i+2]] for i in range(n-1)]
    
    return add_a_char + delete_a_char + revise_a_char + swap_adjacent_two_chars # 暂时返回列表，不返回集合
               
def Identify(words): # 取出words中的词典中存在的正确词
    return set(w for w in words if w[0] in dic)
    
def GetMax(wanteds, n = 3): # 从编辑后的词中选出最多n个置信度最大的
    allwanteds = []
    while len(wanteds) > 0 and n > 0:
        maxword = max(wanteds,key=lambda w : nwords[w[0]]) # 备选词中选出P(w)最大的
        allwanteds.append(maxword)
        wanteds.remove(maxword) 
        n -= 1
    return allwanteds
    
##################### 这里的小写的edit1, edit2, identity, getMax, bayesClassifier 供纠正错词使用

# 一步调整，编辑距离为1
def edit1(word):  
    n = len(word)
    add_a_char = [word[0:i] + c + word[i:] for i in range(n+1) for c in alpha]
    delete_a_char = [word[0:i] + word[i+1:] for i in range(n)]
    revise_a_char = [word[0:i] + c + word[i+1:] for i in range(n) for c in alpha]
    swap_adjacent_two_chars = [word[0:i] + word[i+1] + word[i] + word[i+2:] for i in range(n-1)] 
    return set( add_a_char + delete_a_char +
               revise_a_char +  swap_adjacent_two_chars)

# 两步调整，编辑距离为2
def edit2(word):
    return set(e2 for e1 in edit1(word) for e2 in edit1(e1))

# 朴素贝叶斯分类器
def identify(words): # 取出words中的词典中存在的正确词
    return set(w for w in words if w in dic)

def getMax(wanteds, n = 3): # 从编辑后的词中选出最多n个置信度最大的
    allwanteds = []
    while len(wanteds) > 0 and n > 0:
        maxword = max(wanteds,key=lambda w : nwords[w]) # 备选词中选出P(w)最大的
        allwanteds.append(maxword)
        wanteds.remove(maxword) 
        n -= 1
    return allwanteds

def bayesClassifier(word, n):
    #如果字典中有输入的单词，直接返回
    if identify([word]):
        return [word]
    #一步调整
    wanteds = identify(edit1(word)) 
    if len(wanteds)>0:
        return getMax(wanteds, n)
    #两步调整
    wanteds = identify(edit2(word))
    if len(wanteds)>0:
        return getMax(wanteds, n)
    # 错得太离谱，不再修正
    else:    
        return 

################################# 训练模型 #####################################
# 训练模型时不考虑大小写
training_data1 = brown.words(categories = 'news') + reuters.words()
training_data1 = [word.lower() for word in training_data1]
print('按词读取的语料库读取完毕\n')
training_data2 = brown.sents(categories = 'news') + reuters.sents()
training_data2 = [[word.lower() for word in sents] for sents in training_data2]
print('按句读取的语料库读取完毕\n\n')  # 这里可能耗时较长，一分钟左右

############### noisy channel model ###################
# get P(w)
nwords = collections.Counter(training_data1)
print('P(w)估计完成！\n\n')

# get c(x|w)，把一个正确字符写成错误字符的频数，从error_dict中读取
error_dict = {}
for i in range(len(errors)):
    if errors[i][1] not in error_dict.keys(): # 正确词不在error_dict中
        # {correct:{error:num},...}
        error_dict[errors[i][1]] = {errors[i][0]:error_nums[i]}
    else:  # 因为前面统计错误类型的次数时全部转化为小写了，所以可能有重复，它们算相同的错误
        if errors[i][0] not in error_dict[errors[i][1]].keys():
            error_dict[errors[i][1]][errors[i][0]] = error_nums[i]
        else:
            error_dict[errors[i][1]][errors[i][0]] += error_nums[i]
print('c(x|w)统计完成！\n\n')

# 求混淆矩阵，用字典形式存储
# 1、先统计x/xy出现次数
x_num = dict()
xy_num = dict()
for j, sentence in enumerate(training_data2): # 要考虑各种符号和空格，因此用句子
    for word in sentence: # 句子中的词
        if len(word) == 1: # 这个词只有一个字符
            if word not in x_num.keys():
                x_num[word] = 1
            else:
                x_num[word] += 1
            continue
        for i in range(len(word)): # 词中的字符
            if i <= len(word) - 2: # 倒数第二个字符之前
                if word[i] not in x_num.keys(): # 单字符x
                    x_num[word[i]] = 1
                else:
                    x_num[word[i]] += 1
                if word[i]+word[i+1] not in xy_num.keys(): # 双字符xy
                    xy_num[word[i]+word[i+1]] = 1
                else:
                    xy_num[word[i]+word[i+1]] += 1
            else:  # 为最后一个字符
                if word[i] not in x_num.keys():
                    x_num[word[i]] = 1
                else:
                    x_num[word[i]] += 1
    if j%3000 == 0:
        print('第{}个句子中的字符统计完成'.format(j))
x_num[' '] = len(training_data1) # 以上的统计不包括空格，所以要加进来
print('x,xy次数统计完成\n\n')

# 2、再将c(x|w)分别考虑增删改换除以x_num或者xy_num中的频数得到P(x|w)
# 这里会存在错误统计中的一些元素在训练文本中没出现过的情况，统一频数都取10
for correct in error_dict.keys():
    for error in error_dict[correct].keys():
        if len(correct) == 1 and len(error) == 1: # x打成y
            if error not in x_num:
                error_dict[correct][error] /= 10
            else:
                error_dict[correct][error] /= x_num[error]
                
        if len(correct) == 1 and len(error) == 2: # x打成xy
            if correct not in x_num:
                error_dict[correct][error] /= 10
            else:
                error_dict[correct][error] /= x_num[correct]
        if len(correct) == 2 and len(error) == 1: # xy打成x
            if correct not in xy_num:
                error_dict[correct][error] /= 10
            else:
                error_dict[correct][error] /= xy_num[correct]
        if len(correct) == 2 and len(error) == 2: # xy打成yx
            if correct not in xy_num:
                error_dict[correct][error] /= 10
            else:
                error_dict[correct][error] /= xy_num[correct]
print('混淆矩阵P(x|w)计算完成\n\n')

# 定义专有名词
proper_noun = dict()
proper_noun['fsis'] = 'FSIS'
proper_noun['u.s.'] = 'U.S.'
proper_noun['iranian'] = 'Iranian'
proper_noun['iranians'] = 'Iranians'
proper_noun['hartmarx'] = 'Hartmarx'
proper_noun['brazil'] = 'Brazil'
proper_noun['america'] = 'America'
proper_noun['american'] = 'American'
proper_noun['canada'] = 'Canada'
proper_noun['ltbond'] = 'ltBond'
proper_noun['japan'] = 'Japan'
proper_noun['ltgra'] = 'ltGRA'
proper_noun['cojuangco'] = 'Cojuangco'
proper_noun['shimbun'] = 'Shimbun'
proper_noun['buenos'] = 'Buenos'
proper_noun['hikes'] = 'HIKES'
proper_noun['board'] = 'Board'
proper_noun['sugar'] = 'SUGAR'
proper_noun['trust'] = 'Trust'
proper_noun['expectations'] = 'expectations.'
proper_noun['ltwpm'] = 'ltWPM'
proper_noun['tuesday'] = 'Tuesday'
proper_noun['reuters'] = 'Reuters'
proper_noun['august'] = 'August'
proper_noun['gaf'] = 'GAF'

#################### language model ########################
# 计算unigram, bigram
unigram = []
bigram = []
for i in range(len(training_data2)):
    sentence = training_data2[i]
    
    unis = ngrams(sentence, 1)
    for uni in unis:
        unigram.append(uni)
        
    bis = ngrams(sentence, 2)
    for bi in bis:
        bigram.append(bi)
        
# 单个词出现的频次
unigram = collections.Counter(unigram) 
print('unigram计算完成！\n\n')

# c(wi|wi-1)，知道前一个词，后一个词的频数
bigram = collections.Counter(bigram)
print('bigram计算完成！\n\n')

########################## 利用训练好的模型修改错词并保存改后的句子 ########################
for i in range(len(sentences)):
    Sentence = nltk.word_tokenize(sentences[i]) # 长句分词
    sentence = [] # 小写的句子
    for word in Sentence:
        sentence.append(word.lower())
    mistake = mistakes[i] # 句子中错误个数
    mistake_index = [] # 放句子中错误词的索引
    corrected = [] # 放修改好的词及相应索引
    length = len(sentence)
    ####################### 找到不存在的词并纠正 ##############################
    for j in range(length): # 找到不存在词典中的词
        if sentence[j] not in dic:
            mistake_index.append(j) # 错误词的索引
            mistake -= 1
            
    for k in mistake_index: # 修改不存在的词
        corrects = bayesClassifier(sentence[k], n) # 修改这个错词，有最多n种可行解
        if len(corrects) == 1: # 只有一个满足条件的解
            result = corrects[0]
            corrected.append([result, k])
        else:
            ###################### 加一平滑 ###################
            for correct in corrects:
                if (correct, sentence[k+1]) not in bigram:
                    bigram[(correct, sentence[k+1])] = 1
                else:
                    bigram[(correct, sentence[k+1])] += 1
                if (sentence[k-1], correct) not in bigram:
                    bigram[(sentence[k-1], correct)] = 1
                else:
                    bigram[(sentence[k-1], correct)] += 1

            ##################### 计算单词概率 P(wi-1,wi)P(wi,wi+1)最大的词 #################
            if k == 0: # 单词在句首
                possibilities = [bigram[(correct, sentence[k+1])] for correct in corrects]
            elif k == length - 1: # 单词在句尾
                possibilities = [bigram[(sentence[k-1], correct)] for correct in corrects]
            else: # P(wi,wi-1)P(wi+1,wi)
                possibilities = [bigram[(sentence[k-1], correct)]*bigram[(correct, sentence[k+1])] 
                                 for correct in corrects]
            max_p = max(possibilities) #备选词中概率最高的
            index = possibilities.index(max_p)
            result = corrects[index]
            corrected.append([result, k]) # [修改后的词，原句索引]

    ######################## 找到可能打错的真词 ################################
    if mistake != 0: # 除了不存在的词以外还有错词
        wrong_possibility = [] # 每个词错误的概率，用改成最佳候选词的效益来表示
        for j in range(len(sentence)): # 选出正确可能性最低的mistake个词
            word = sentence[j]
            if j in mistake_index:  # 不考虑改过的拼错词
                continue
            alternatives = GetMax(Edit(word), m) # m个备选词（可能包括自己）
            if word not in alternatives:
                alternatives.append([word,'',''])
            confidence = [] # 置信度
            for alt in alternatives: # 逐一计算备选词的可信度
                # 没有出现过的次数就按照0算
                if alt[1] not in error_dict.keys():
                    error_dict[alt[1]] = {} 
                    error_dict[alt[1]][alt[2]] = 0
                elif alt[2] not in error_dict[alt[1]].keys():
                    error_dict[alt[1]][alt[2]] = 0
                p = nwords[alt[0]] * error_dict[alt[1]][alt[2]] # P(w)P(x|w)
                confidence.append([alt[0], p]) # 改后每个词以及它的概率
            alternative = sorted(confidence, key = lambda x: x[1])[-1] # 置信度最高的备选词及它的概率
            p_self = nwords[word] * 0.95 # P(x)P(x|x)，在这里P(x|x)认为是0.95
            # 改成最佳候选词的效益（错误率），这个词, 索引
            wrong_possibility.append([alternative[1] - p_self, alternative[0], j]) 
        # 错误率从低到高排序，还有几个错就选几个词改正
        wrong_words = sorted(wrong_possibility, key = lambda x: x[0])[-mistake:] 
        
        ######################## 纠正可能打错的真词 ################################
        mistake_index = [wrong_word[2] for wrong_word in wrong_words]
        for k in mistake_index: # 修改真词
            corrects = bayesClassifier(sentence[k], n) # 修改这个词，有最多n种可行解
            if len(corrects) == 1: # 只有一个满足条件的解
                result = corrects[0]
                corrected.append([result, k])
            else:
                ###################### 加一平滑 ###################
                for correct in corrects:
                    if (correct, sentence[k+1]) not in bigram:
                        bigram[(correct, sentence[k+1])] = 1
                    else:
                        bigram[(correct, sentence[k+1])] += 1
                    if (sentence[k-1], correct) not in bigram:
                        bigram[(sentence[k-1], correct)] = 1
                    else:
                        bigram[(sentence[k-1], correct)] += 1

                ##################### 计算单词概率 P(wi-1,wi)P(wi,wi+1)最大的词 #################
                if k == 0: # 单词在句首
                    possibilities = [bigram[(correct, sentence[k+1])] for correct in corrects]
                elif k == length - 1: # 单词在句尾
                    possibilities = [bigram[(sentence[k-1], correct)] for correct in corrects]
                else: # P(wi,wi-1)P(wi+1,wi)
                    possibilities = [bigram[(sentence[k-1], correct)]*bigram[(correct, sentence[k+1])] 
                                     for correct in corrects]
                max_p = max(possibilities) #备选词中概率最高的
                index = possibilities.index(max_p)
                result = corrects[index]
                corrected.append([result, k]) # [修改后的词，原句索引]
    
    ############################ 将找到的纠正词用于纠正原句子，并输出到文本 #########################
    for j in range(len(corrected)):
        if corrected[j][0] in proper_noun.keys(): # 专有名词有自己的改法
            Sentence[corrected[j][1]] = proper_noun[corrected[j][0]]
            continue
        if len(Sentence[corrected[j][1]]) >= 2:
            if Sentence[corrected[j][1]][0].isupper() and Sentence[corrected[j][1]][1].isupper(): # 全大写句子
                Sentence[corrected[j][1]] = corrected[j][0].upper()
                continue
        if Sentence[corrected[j][1]][0].isupper(): # 普通句子，要改正的词第一个字母是大写的
            Sentence[corrected[j][1]] = corrected[j][0].capitalize() # 首字母转换大写
        else: # 普通词语，小写开头
            Sentence[corrected[j][1]] = corrected[j][0]
    correct_sent = ' '.join(Sentence) + '\n'
    
    with open('result.txt','a') as f:
        f.write('{}\t'.format(i+1) + correct_sent)
        
    if (i+1) % 50 == 0:
        print('第{}个纠正后的句子写入完毕'.format(i+1))
 