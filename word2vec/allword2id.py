# coding=UTF-8  

"""
# @File  : word2id.py
# @Author: HM
# @Date  : 2018/5/2
将all_word_list转变成allword2id
"""

wordList=[]
with open("all_word_list.txt",encoding="UTF-8",mode="r+") as f:
    for line in f:
        line = line.replace("\n", "")
        wordList.append(line)
f.close()

with open("allword2id.txt",encoding="UTF-8",mode="w+") as f:
    i=0
    for word in wordList:
        f.write(str(i)+" ")
        f.write(word+"\n")
        i+=1
f.close()

