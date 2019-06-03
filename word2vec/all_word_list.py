# coding=UTF-8  

"""
# @File  : all_word_list.py
# @Author: HM
# @Date  : 2018/4/26
对所有数据 提取出所有的词汇表
"""
import re
import os
from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.config import FLAGS

catalog = FLAGS.catalog + "dataset/"
all_word_list_f = FLAGS.catalog + "word2vec/dataset_all_word.txt"

# file_list = ["books/labeled_neg.txt", "books/labeled_pos.txt",
#              "books/unlabeled_neg.txt",   "books/unlabeled_pos.txt",
#              "dvd/labeled_neg.txt", "dvd/labeled_pos.txt",
#              "dvd/unlabeled_neg.txt",   "dvd/unlabeled_pos.txt",
#              "electronics/labeled_neg.txt", "electronics/labeled_pos.txt",
#              "electronics/unlabeled_neg.txt",   "electronics/unlabeled_pos.txt",
#              "kitchen/labeled_neg.txt", "kitchen/labeled_pos.txt",
#              "kitchen/unlabeled_neg.txt",   "kitchen/unlabeled_pos.txt"]
file_list = ["books/review_negative", "books/review_positive",
             "books/review_unlabeled",
             "dvd/review_negative", "dvd/review_positive",
             "dvd/review_unlabeled",
             "electronics/review_negative", "electronics/review_positive",
             "electronics/review_unlabeled",
             "kitchen/review_negative", "kitchen/review_positive",
             "kitchen/review_unlabeled"]

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")


def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, " ", string.lower())


def make_word_list(file, list):
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


def generate_word_l(file_list, word_list_f):
    list = []
    if os.path.exists(word_list_f):
        with open(word_list_f, encoding="UTF-8", mode="r+") as rf:
            for i in rf:
                i = i.replace("\n", "")
                list.append(i)
        rf.close()
    for i, file in enumerate(file_list):
        list = make_word_list(catalog + file, list)
        print("the length of the word_list is %d" % len(list))
        with open("dataset_all_word.txt", encoding="UTF-8", mode="w+") as f_dw:
            for word in list:
                f_dw.write(word + '\n')
        f_dw.close()


def main():
    generate_word_l(file_list, all_word_list_f)


if __name__ == '__main__':
    main()
