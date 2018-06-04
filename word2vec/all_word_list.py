# coding=UTF-8  

"""
# @File  : all_word_list.py
# @Author: HM
# @Date  : 2018/4/26
对所有数据 提取出所有的词汇表
"""
import re

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")


def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, " ", string.lower())

def make_word_list(file, list):
    num = 0
    with open(file, encoding="UTF-8", mode="r+") as f:
        for line in f:
            line = cleanSentences(line)
            line = line.strip()
            split = line.split()
            for word in split:
                if word not in list:
                    list.append(word)
    f.close()
    print("file {} finished!".format(file))
    return list



file_list = ["../data/books/negative_processed", "../data/books/positive_processed",
             "../data/dvd/negative_processed",   "../data/dvd/positive_processed",
             "../data/electronics/negative_processed", "../data/electronics/positive_processed",
             "../data/kitchen_&_housewares/negative_processed", "../data/kitchen_&_housewares/positive_processed",
             "../data/books/unlabeled_negative", "../data/books/unlabeled_positive",
             "../data/dvd/unlabeled_negative",  "../data/dvd/unlabeled_positive",
             "../data/electronics/unlabeled_negative", "../data/electronics/unlabeled_positive",
             "../data/kitchen_&_housewares/unlabeled_negative", "../data/kitchen_&_housewares/unlabeled_positive"
             ]

list=[]
for file in file_list:
    list=make_word_list(file, list)
    print("the length of the word_list is %d"%len(list))
with open("all_word_list.txt", encoding="UTF-8", mode="w+") as f_dw:
    for word in list:
        f_dw.write(word+ '\n')
f_dw.close()