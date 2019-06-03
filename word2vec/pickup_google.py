# coding=UTF-8  

"""
# @File  : DAS_pickup_google.py
# @Author: HM
# @Date  : 2018/4/26
从google预训练的词向量找到需要的

"""

import numpy as np
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("word_dim", 300, "wordvec dimension")
FLAGS = flags.FLAGS

"""1.查找google new训练的词向量中存在与wordlist的词向量"""


def lookup_word_vector(word_list_f, google_word2vec_f, wordvec_f):
    wordList = []  # 存放单词
    with open(word_list_f, encoding="UTF-8", mode="r+") as f:
        for word in f:
            word = word.replace("\n", "")
            wordList.append(word)
    f.close()
    print("load wordlist finished!")

    wordVec = []
    i = 0
    print(wordList)
    with open(google_word2vec_f, encoding="UTF-8", mode="r") as f_goo:
        for line in f_goo:
            i += 1
            if i == 1:
                continue
            else:
                line = line.replace("\n", "")
                value = line.split()
                goo_word = value[0]
                if goo_word in wordList:
                    wordVec.append(line)
                    print("add...")
            print("i=%d" % i)

    f.close()
    print("load google word2vec finished!")
    print("length is ")
    print(len(wordVec))  # 61688
    np.save(wordvec_f, wordVec)


"""2.生成词向量矩阵"""


def generate_matrix(word_list_f, wordvec_f, embedding_matrix_f):
    wordList_dic = {}
    with open(word_list_f, encoding="UTF-8", mode="r") as f:
        for line in f:
            line = line.replace("\n", "")
            wordList_dic[line] = len(wordList_dic)
    f.close()
    print("load wordlist_dic finished!")
    print("the length od wordlist_dic:%d" % len(wordList_dic))

    # 加载google news中存在于实验数据的词向量
    wordVec = np.load(wordvec_f + ".npy")
    print(wordVec.shape)  # 输出： 多少词
    embeddings_index = {}  # 存放词对应 词向量的字典 将wordVec转化成字典类型
    for line in wordVec:
        line = line.replace("\n", "")
        value = line.split()
        word = value[0]
        vector = np.asarray(value[1:], dtype='float32')
        embeddings_index[word] = vector

    print("found %s word vector." % len(embeddings_index))  # 与wordVec.shape的数应该一样

    print("Preparing embedding matrix.")
    embedding_matrix = np.zeros([len(wordList_dic) + 1, FLAGS.word_dim])

    # 把所有存在于wordVec里的词向量拿出来，存成index-vector的embedding_matrix字典
    for (word, i) in wordList_dic.items():
        if i > len(wordVec):  # 如果不在wordVec里，跳过，就为初始化的零向量
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("Training weight matrix.")
    np.save(embedding_matrix_f, embedding_matrix)


"""3.查看embedding_matrix"""


def print_matrix_size(embedding_matrix_f):
    embedding_matrix = np.load(embedding_matrix_f + ".npy")
    print(embedding_matrix)
    print(embedding_matrix.shape)  # (117040, 300) ->(118925,300)


def main():
    word_list_f = "dataset_all_word.txt"
    google_word2vec_f = "googleNews-vectors-negative300.txt"
    wordvec_f = "dataset_allwordVec"
    embedding_matrix_f = "dataset_all_embedding_matrix"
    lookup_word_vector(word_list_f, google_word2vec_f, wordvec_f)
    generate_matrix(word_list_f, wordvec_f, embedding_matrix_f)


if __name__ == "__main__":
    main()
