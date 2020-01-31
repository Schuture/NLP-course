'''# min edit distance 希望找到两个词之间最小的编辑距离
# 动态规划，与求最长公共子序列相同
# 输入为两个词x,y，输出例如：4

思路：
用dp[i][j]来存放word1前i个字符和word2前j个字符的编辑距离
1 假如word1和word2最后一个字符相同，那么去掉这两个字符，编辑距离不改变dp[i][j] = dp[i-1][j-1]
2 假如word1和word2最后一个字符不相同，那么有三种可能性
    1) word1[:i-1]要与word2[:j]相等，得word1增加一个字符
    2) word1[:i]要与word2[:j-1]相等，得word2增加一个字符
    3) word1[:i]要与word2[:j]相等，得替换最后一个字符
'''
x,y = 'intention','execution' 
def minDistance(word1, word2):
    '''每一步有三种可能性：两个词长度都+1，word1 + 1, word2 + 1'''
    m, n = len(word1), len(word2)
    dp = [[0 for j in range(n+1)] for i in range(m+1)] # m+1行，n+1列
    # initialization
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    # 迭代构建dp
    for i in range(1,m+1):
        for j in range(1,n+1): 
            minimum = min([dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1] if \
                           word1[i-1]==word2[j-1] else dp[i-1][j-1]+1])
            dp[i][j] = minimum
    return dp[m][n] # 两个完整词的最小编辑距离
    
print(minDistance(x,y))