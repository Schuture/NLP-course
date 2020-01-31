voc = ['我','我们','爱','自然','语言','自然语言','自然语言处理','处理']
string = '我们爱自然语言处理'

def is_prefix(string, voc): # 判断string是否为voc中元素的前缀
    for word in voc:
        if string==word[:len(string)]:
            return True
    return False
            
begin,end = 0,0 # 截取字符串包括首尾
ret = ''
while end<=len(string):
    if end==len(voc):
        ret += string[begin:]
    if is_prefix(string[begin:end+1], voc):
        end += 1
    else: # 在第一个找不到前缀配对的位置将前一个词输出
        ret += string[begin:end] + ' '
        begin = end
        
print(ret)