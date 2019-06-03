# coding=UTF-8  

"""
# @File  : word2id.py
# @Author: HM
# @Date  : 2018/5/2
将all_word_list转变成allword2id
"""

word_list_f="dataset_all_word.txt"
word2id_f="dataset_allword2id.txt"


def generate_word2id(word_list_f,word2id_f):
    wordList=[]
    with open(word_list_f,encoding="UTF-8",mode="r+") as f:
        for line in f:
            line = line.replace("\n", "")
            wordList.append(line)
    f.close()

    with open(word2id_f,encoding="UTF-8",mode="w+") as f:
        i=0
        for word in wordList:
            f.write(str(i)+" ")
            f.write(word+"\n")
            i+=1
    f.close()

def main():
    generate_word2id(word_list_f, word2id_f)

if __name__ == '__main__':
    main()



