# coding=UTF-8  

"""
# @File  : decide_max_seq.py
# @Author: HM
# @Date  : 2018/4/26
确定max_seq
"""
import re

"""1、对所有数据统计每条评论的单词数 存为word_num"""
# strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
#
#
# def cleanSentences(string):
#     string = string.lower().replace("<br />", " ")
#     return re.sub(strip_special_chars, " ", string.lower())
#
#
# def print_word_num(file, list):
#     num = 0
#     with open(file, encoding="UTF-8", mode="r+") as f:
#         for line in f:
#             line = cleanSentences(line)
#             line = line.strip()
#             split = line.split()
#             list.append(len(split))
#     print("finish+{}".format(file))
#
#
# file_list = ["../data/books/negative_processed", "../data/books/positive_processed",
#              "../data/dvd/negative_processed",   "../data/dvd/negative_processed",
#              "../data/electronics/negative_processed", "../data/electronics/negative_processed",
#              "../data/kitchen_&_housewares/negative_processed", "../data/kitchen_&_housewares/negative_processed",
#              "../data/books/unlabeled_negative", "../data/books/unlabeled_positive",
#              "../data/dvd/unlabeled_negative",  "../data/dvd/unlabeled_positive",
#              "../data/electronics/unlabeled_negative", "../data/electronics/unlabeled_positive",
#              "../data/kitchen_&_housewares/unlabeled_negative", "../data/kitchen_&_housewares/unlabeled_positive"
#              ]
# list=[]
# for file in file_list:
#     print_word_num(file,list)
# with open("word_num.txt", encoding="UTF-8", mode="w+") as f_dw:
#     for num in list:
#         f_dw.write(str(num)+ '\n')
# f_dw.close()


"""2、从文件中直接读取1生成的wordnum文件 来确定最大单词数设为多少"""
list = []
with open("word_num.txt", encoding="UTF-8", mode="r+") as f_dw:
    for line in f_dw:
        line = line.replace("\n", "")
        list.append(line)
f_dw.close()
fifth = 0
eighth = 0
onehunred = 0
onefive = 0
twohunred = 0
twofive = 0
threehunred = 0
total=0
for i in list:
    total+=int(i)
    if int(i) > 50:
        fifth += 1
    if int(i) > 80:
        eighth += 1
    if int(i) > 100:
        onehunred += 1
    if int(i) > 150:
        onefive += 1
    if int(i) > 200:
        twohunred += 1
    if int(i) > 250:
        twofive += 1
    if int(i) > 300:
        threehunred += 1
print("fifth:%d,eighth:%d,onehunred:%d,onefive:%d,twohunred:%d,twofive:%d,threehunred:%d" % (
fifth, eighth, onehunred, onefive, twohunred, twofive, threehunred))
print("total:%d"%len(list))
print("average word num per sentence:%d"%(total/len(list)))
# fifth: 58208, eighth: 43902, onehunred: 36386, onefive: 23742,
# twohunred: 16439, twofive: 11890, threehunred: 8913
# total: 78679
# average word num per sentence:145