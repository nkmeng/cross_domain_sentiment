# coding=UTF-8  

"""
# @File  : data_helper.py
# @Author: HM
# @Date  : 2018/4/4
"""
import numpy as np
import pickle
import os
import nltk
import random
import re

# 删除标点符号，括号，问号等，只留下字母数字字符
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")


def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, " ", string.lower())


def shuffle_test(x, y, x_trans):
    data_len = len(x)
    data = np.array(x)
    label = np.array(y)
    tag_x = np.array(x_trans)

    shuffle_idx = np.random.permutation(np.arange(data_len))  # 对data_len长度得数组打乱顺序
    shuffle_data = data[shuffle_idx]
    shuffle_label = label[shuffle_idx]
    shuffle_tag_x = tag_x[shuffle_idx]
    return shuffle_data, shuffle_label, shuffle_tag_x


def batches_iter(data, label, tag_x, batch_size, shuffle=True):
    """
    iterate the data
    """
    data_len = len(data)
    batch_num = int(data_len / batch_size)  # 一个batch中多少data
    data = np.array(data)
    label = np.array(label)
    tag_x = np.array(tag_x)

    if shuffle:  # 将所有元素随机排序
        shuffle_idx = np.random.permutation(np.arange(data_len))  # 对data_len长度得数组打乱顺序
        shuffle_data = data[shuffle_idx]
        shuffle_label = label[shuffle_idx]
        shuffle_tag_x = tag_x[shuffle_idx]
    else:
        shuffle_data = data
        shuffle_label = label
        shuffle_tag_x = tag_x

    for batch in range(batch_num):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, data_len)
        yield shuffle_data[start_idx: end_idx], shuffle_label[start_idx:end_idx], shuffle_tag_x[start_idx:end_idx]


def joint_batches_iter(x, y, tag_x, x_trans, y_trans, p_pivots_trans, n_pivots_trans, batch_size, shuffle=True):
    """
    iterate the data
    """
    data_len = len(x)
    batch_num = int(data_len / batch_size)  # 一个batch中多少data
    x = np.array(x)
    y = np.array(y)
    tag_x = np.array(tag_x)
    x_trans = np.array(x_trans)
    y_trans = np.array(y_trans)
    p_pivots_trans = np.array(p_pivots_trans)
    n_pivots_trans = np.array(n_pivots_trans)

    if shuffle:  # 将所有元素随机排序
        shuffle_idx = np.random.permutation(np.arange(data_len))  # 对data_len长度得数组打乱顺序
        shuffle_x = x[shuffle_idx]
        shuffle_y = y[shuffle_idx]
        shuffle_tag_x = tag_x[shuffle_idx]
        shuffle_x_trans = x_trans[shuffle_idx]
        shuffle_y_trans = y_trans[shuffle_idx]
        shuffle_p_pivots_trans = p_pivots_trans[shuffle_idx]
        shuffle_n_pivots_trans = n_pivots_trans[shuffle_idx]
    else:
        shuffle_x = x
        shuffle_y = y
        shuffle_tag_x = tag_x
        shuffle_x_trans = x_trans
        shuffle_y_trans = y_trans
        shuffle_p_pivots_trans = p_pivots_trans
        shuffle_n_pivots_trans = n_pivots_trans

    for batch in range(batch_num):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, data_len)
        yield shuffle_x[start_idx: end_idx], shuffle_y[start_idx:end_idx], \
              shuffle_tag_x[start_idx:end_idx], shuffle_x_trans[start_idx:end_idx], \
              shuffle_y_trans[start_idx:end_idx], shuffle_p_pivots_trans[start_idx:end_idx], \
              shuffle_n_pivots_trans[start_idx:end_idx]


def trainTestSplit(source_x, source_doc_y, source_tag_x, test_size=0.2):
    X_num = source_x.shape[0]
    train_index_p = range(0, X_num // 2)
    train_index_n = range(X_num // 2, X_num)
    train_index = []
    test_num = int(X_num * test_size)

    p_choice = random.sample(train_index_p, test_num // 2)
    n_choice = random.sample(train_index_n, test_num // 2)
    test_index = p_choice + n_choice
    for i in range(X_num):
        if i not in test_index:
            train_index.append(i)
    print(len(train_index))

    train_x, valid_x, train_y, valid_y, train_tag_x, valid_tag_x, train_x_trans, valid_x_trans, train_y_trans, valid_y_trans, train_x_pivots_trans, valid_x_pivots_trans = [], [], [], [], [], [], [], [], [], [], [], []
    for i, data in enumerate(source_x):
        if i in train_index:
            train_x.append(data)
        elif i in test_index:
            valid_x.append(data)
        else:
            print("都不在？")
    for i, data in enumerate(source_doc_y):
        if i in train_index:
            train_y.append(data)
        elif i in test_index:
            valid_y.append(data)
        else:
            print("都不在？")
    for i, data in enumerate(source_tag_x):
        if i in train_index:
            train_tag_x.append(data)
        elif i in test_index:
            valid_tag_x.append(data)
        else:
            print("都不在？")
    return np.asarray(train_x), np.asarray(valid_x), np.asarray(train_y), np.asarray(valid_y), np.asarray(
        train_tag_x), np.asarray(valid_tag_x)


def joint_trainTestSplit(source_x, source_doc_y, source_tag_x, source_x_trans, source_doc_y_trans,
                         source_p_pivots_trans, source_n_pivots_trans, test_size=0.2):
    X_num = source_x.shape[0]
    train_index_p = range(0, X_num // 2)
    train_index_n = range(X_num // 2, X_num)
    train_index = []
    test_num = int(X_num * test_size)

    p_choice = random.sample(train_index_p, test_num // 2)
    n_choice = random.sample(train_index_n, test_num // 2)
    test_index = p_choice + n_choice
    for i in range(X_num):
        if i not in test_index:
            train_index.append(i)
    print(len(train_index))

    train_x, valid_x, train_y, valid_y, train_tag_x, valid_tag_x, train_x_trans, valid_x_trans, train_y_trans, valid_y_trans, train_p_pivots_trans, valid_p_pivots_trans, train_n_pivots_trans, valid_n_pivots_trans = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for i, data in enumerate(source_x):
        if i in train_index:
            train_x.append(data)
        elif i in test_index:
            valid_x.append(data)
        else:
            print("都不在？")
    for i, data in enumerate(source_doc_y):
        if i in train_index:
            train_y.append(data)
        elif i in test_index:
            valid_y.append(data)
        else:
            print("都不在？")
    for i, data in enumerate(source_tag_x):
        if i in train_index:
            train_tag_x.append(data)
        elif i in test_index:
            valid_tag_x.append(data)
        else:
            print("都不在？")
    for i, data in enumerate(source_x_trans):
        if i in train_index:
            train_x_trans.append(data)
        elif i in test_index:
            valid_x_trans.append(data)
        else:
            print("都不在？")
    for i, data in enumerate(source_doc_y_trans):
        if i in train_index:
            train_y_trans.append(data)
        elif i in test_index:
            valid_y_trans.append(data)
        else:
            print("都不在？")
    for i, data in enumerate(source_p_pivots_trans):
        if i in train_index:
            train_p_pivots_trans.append(data)
        elif i in test_index:
            valid_p_pivots_trans.append(data)
        else:
            print("都不在？")
    for i, data in enumerate(source_n_pivots_trans):
        if i in train_index:
            train_n_pivots_trans.append(data)
        elif i in test_index:
            valid_n_pivots_trans.append(data)
        else:
            print("都不在？")
    return np.asarray(train_x), np.asarray(valid_x), np.asarray(train_y), np.asarray(valid_y), np.asarray(
        train_tag_x), np.asarray(valid_tag_x), np.asarray(train_x_trans), np.asarray(valid_x_trans), np.asarray(
        train_y_trans), np.asarray(valid_y_trans), np.asarray(train_p_pivots_trans), np.asarray(
        valid_p_pivots_trans), np.asarray(train_n_pivots_trans), np.asarray(valid_n_pivots_trans)


def set_file(src_domain, tar_domain, FLAGS):
    if src_domain == "books":
        src_file_neg = FLAGS.books_neg
        src_file_pos = FLAGS.books_pos
    elif src_domain == "dvd":
        src_file_neg = FLAGS.dvd_neg
        src_file_pos = FLAGS.dvd_pos
    elif src_domain == "ele":
        src_file_neg = FLAGS.ele_neg
        src_file_pos = FLAGS.ele_pos
    else:
        src_file_neg = FLAGS.kit_neg
        src_file_pos = FLAGS.kit_pos

    if tar_domain == "books":
        tar_file_neg = FLAGS.books_u_neg
        tar_file_pos = FLAGS.books_u_pos
    elif tar_domain == "dvd":
        tar_file_neg = FLAGS.dvd_u_neg
        tar_file_pos = FLAGS.dvd_u_pos
    elif tar_domain == "ele":
        tar_file_neg = FLAGS.ele_u_neg
        tar_file_pos = FLAGS.ele_u_pos
    else:
        tar_file_neg = FLAGS.kit_u_neg
        tar_file_pos = FLAGS.kit_u_pos
    return src_file_pos, src_file_neg, tar_file_pos, tar_file_neg


def make_str(file):
    if "dvd" in file:
        fstr = "dvd"
    elif "book" in file:
        fstr = "book"
    elif "ele" in file:
        fstr = "ele"
    else:
        fstr = "kit"
    if "positive" in file:
        fstr += "_pos"
    else:
        fstr += "neg"
    return fstr


# 测试和域训练的目标域数据 分开取，取的数目为num
def generate_random(num, file, ):
    with open(file, encoding="UTF-8", mode="r+") as f:
        count = -1
        for count, line in enumerate(f):
            pass
        count += 1
        print("the num of data in the file:%d" % count)

    if num == count:
        print("the num of choice is equal to the file num")
        print("all add to load!")
        choice = list(range(0, num))
        print(choice)
    elif num < count:
        print("the num of choice is less than the file num")
        print("choose randomly to load!")
        randomdata = range(0, count)
        choice = random.sample(randomdata, num)
    else:
        print("the num of choice is more than the file num")
        print("是因为测试数据和目标域中用于训练的数据不能完全不重合")
        return
        # random_test = random.sample(range(0, count), FLAGS.test_num)
        # random_train = random.sample(range(0, count), num - FLAGS.test_num)
        # choice = random_test + random_train
    fstr = make_str(file)
    with open("../save/randomdata/"+fstr + str(num) + ".txt", mode="a+", encoding="UTF-8") as f1:
        for i in choice:
            f1.write(str(i) + "\n")
    f.close()
    f1.close()

    return choice


def load_p_pivots_list(src_domain, tar_domain):
    pivots_list = []
    with open("../save/pivots/" + src_domain + "_" + tar_domain + "_pos.txt") as f:
        for line in f:
            line = line.replace("\n", "")
            pivots_list.append(line)
    f.close()
    return pivots_list


def load_n_pivots_list(src_domain, tar_domain):
    pivots_list = []
    with open("../save/pivots/" + src_domain + "_" + tar_domain + "_neg.txt") as f:
        for line in f:
            line = line.replace("\n", "")
            pivots_list.append(line)
    f.close()
    return pivots_list


def load_word2id(word2id_file, encoding='UTF-8'):
    """
    :param word2id_file: word-id mapping file path
    :param encoding: file's encoding,for changing to unicode
    :return: word-id mapping,like hello=5
    """
    word2id = dict()
    with open(word2id_file, encoding="UTF-8", mode="r+") as f:
        for line in f:
            l = line.split()
            word2id[l[1]] = int(l[0])
    f.close()
    print('\nload word-id mapping done！\n')
    return word2id


def load_id2word(word2id_file, encoding='UTF-8'):
    """
    :param word2id_file: word-id mapping file path
    :param encoding: file's encoding,for changing to unicode
    :return: word-id mapping,like hello=5
    """
    id2word = dict()
    with open(word2id_file, encoding="UTF-8", mode="r+") as f:
        for line in f:
            l = line.split()
            id2word[float(l[0])] = l[1]
    f.close()
    print('\nload id-word mapping done！\n')
    return id2word


def transform_data(words, p_pivots_list, n_pivots_list):
    pnum = 0
    nnum = 0
    for index, word in enumerate(words):
        if word in p_pivots_list:
            words[index] = "UNK"
            pnum += 1
    for index, word in enumerate(words):
        if word in n_pivots_list:
            words[index] = "UNK"
            nnum += 1
    return words, pnum, nnum


def load_src(file, word_to_id, max_seq, FLAGS, type=None, encoding='UTF-8', transform=False):
    x, tag_x, p_pivots, n_pivots = [], [], [], []

    cixing = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", ]  # "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"

    if transform:
        p_pivots_list = load_p_pivots_list(FLAGS.src_domain, FLAGS.tar_domain)
        n_pivots_list = load_n_pivots_list(FLAGS.src_domain, FLAGS.tar_domain)
    for l1 in file:
        p_pivots_num = 0
        n_pivots_num = 0
        t_x = np.zeros(max_seq)  # 存放数据，值为id,
        tag = np.zeros(max_seq)  # 存放是否为需要词性，是为1，不是为0
        j = 0  # 记录一句话有几个单词
        sentence = cleanSentences(l1)
        sentence = sentence.strip()
        sentence = sentence.replace("\n", "")
        if sentence == "":
            pass
        else:
            words = sentence.split()
            if transform:
                words_p, p_pivots_num, n_pivots_num = transform_data(words, p_pivots_list, n_pivots_list)
            pos_tag = nltk.pos_tag(words)
            for word in pos_tag:
                if j < max_seq:
                    if word[0] in word_to_id:  # 单词在word_to_id里，
                        t_x[j] = word_to_id[word[0]]
                        if word[1] in cixing:
                            tag[j] = 1
                        else:
                            tag[j] = 0
                    elif word[0] == "UNK":
                        t_x[j] = len(word_to_id)
                        tag[j] = 0
                    else:
                        print("词是：%s 不在word_to_id里" % word)
                    j += 1
                else:  # 超过最大单词数截断
                    break

        x.append(t_x)
        tag_x.append(tag)
        if p_pivots_num > 0:
            p_pivots.append([1., 0.])
        else:
            p_pivots.append([0., 1.])
        if n_pivots_num > 0:
            n_pivots.append([1., 0.])
        else:
            n_pivots.append([0., 1.])

    print("the length of x:%d" % len(x))
    return np.asarray(x), np.asarray(tag_x), np.asarray(p_pivots), np.asarray(n_pivots)


def src_data(pfile, nfile, word_to_id, max_seq, FLAGS, shuffle=False, transform=False):
    with open(pfile, encoding="UTF-8", mode="r+") as f1:
        print("\nload positive file:{}".format(pfile))
        x_pos, tag_x_pos, p_pivots_pos, n_pivots_pos = load_src(f1, word_to_id, max_seq, FLAGS, transform=transform)
        y_pos = []
        for i in range(len(x_pos)):
            y_pos.append([1., 0.])

    with open(nfile, encoding="UTF-8", mode="r+") as f2:
        print("\nload negative file:{}".format(nfile))
        x_neg, tag_x_neg, p_pivots_neg, n_pivots_neg = load_src(f2, word_to_id, max_seq, FLAGS, transform=transform)
        y_neg = []
        for i in range(len(x_neg)):
            y_neg.append([0., 1.])

    if (shuffle):  # 一积一消
        x, y, tag_x, p_pivots, n_pivots = [], [], [], [], []
        pos_num = len(x_pos)
        neg_num = len(x_neg)
        m, n = 0, 0

        for i in range(pos_num + neg_num):
            if i % 2 == 0:
                x.append(x_pos[m])
                y.append(y_pos[m])
                tag_x.append(tag_x_pos[m])
                p_pivots.append(p_pivots_pos[m])
                n_pivots.append(n_pivots_pos[m])
                m += 1

            else:
                x.append(x_neg[n])
                y.append(y_neg[n])
                tag_x.append(tag_x_neg[n])
                p_pivots.append(p_pivots_neg[n])
                n_pivots.append(n_pivots_neg[n])
                n += 1

    else:  # 先积后消
        x, y, tag_x, p_pivots, n_pivots = [], [], [], [], []

        for i in range(len(x_pos)):
            x.append(x_pos[i])
            y.append(y_pos[i])
            tag_x.append(tag_x_pos[i])
            p_pivots.append(p_pivots_pos[i])
            n_pivots.append(n_pivots_pos[i])
        for i in range(len(x_neg)):
            x.append(x_neg[i])
            y.append(y_neg[i])
            tag_x.append(tag_x_neg[i])
            p_pivots.append(p_pivots_neg[i])
            n_pivots.append(n_pivots_neg[i])

    if transform:

        fw = open("../save/datafile/"+FLAGS.src_domain+"_src_trans.txt", 'wb')
        pickle.dump(np.asarray(x), fw, -1)  # Pickle dictionary using protocol 0.
        pickle.dump(np.asarray(y), fw)
        pickle.dump(np.asarray(tag_x), fw)
        pickle.dump(np.asarray(p_pivots), fw)
        pickle.dump(np.asarray(n_pivots), fw)

    else:
        fw = open("../save/datafile/"+FLAGS.src_domain+"_src.txt", 'wb')
        pickle.dump(np.asarray(x), fw, -1)  # Pickle dictionary using protocol 0.
        pickle.dump(np.asarray(y), fw)
        pickle.dump(np.asarray(tag_x), fw)


def load_tar(choice, file, word_to_id, max_seq, FLAGS, type=None, encoding='UTF-8', transform=False):
    x, tag_x, p_pivots, n_pivots = [], [], [], []
    pickup_line = []
    cixing = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", ]  # "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"

    if transform:
        p_pivots_list = load_p_pivots_list(FLAGS.src_domain, FLAGS.tar_domain)
        n_pivots_list = load_n_pivots_list(FLAGS.src_domain, FLAGS.tar_domain)

    for findex, l1 in enumerate(file):

        if findex in choice:
            pickup_line.append(l1)
            t_x = np.zeros(max_seq)  # 存放数据，值为id,
            tag = np.zeros(max_seq)  # 存放是否为需要词性，是为1，不是为0
            p_pivots_num = 0
            n_pivots_num = 0
            j = 0  # 计sentence里的word数
            sentence = cleanSentences(l1)
            sentence = sentence.strip()
            sentence = sentence.replace("\n", "")
            if sentence == "":
                pass
            else:
                words = sentence.split()
                if transform:
                    words, p_pivots_num, n_pivots_num = transform_data(words, p_pivots_list, n_pivots_list)
                pos_tag = nltk.pos_tag(words)
                for word in pos_tag:
                    if j < max_seq:
                        if word[0] in word_to_id:  # 单词在word_to_id里，
                            t_x[j] = word_to_id[word[0]]
                            if word[1] in cixing:
                                tag[j] = 1
                            else:
                                tag[j] = 0
                        elif word[0] == "UNK":
                            t_x[j] = len(word_to_id)
                            tag[j] = 0
                        else:
                            print("词是：%s 不在word_to_id里" % word)
                        j += 1
                    else:  # 超过最大单词数截断
                        break

            x.append(t_x)
            tag_x.append(tag)
            if p_pivots_num > 0:
                p_pivots.append([1., 0.])
            else:
                p_pivots.append([0., 1.])
            if n_pivots_num > 0:
                n_pivots.append([1., 0.])
            else:
                n_pivots.append([0., 1.])

    print("the length of x:%d" % len(x))

    return np.asarray(x), np.asarray(tag_x), np.asarray(p_pivots), np.asarray(n_pivots), pickup_line


def tar_data(num, data_pos, data_neg, word_to_id, max_seq, FLAGS, shuffle=False, transform=False):
    with open(data_pos, encoding="UTF-8", mode="r+") as f1:
        print("\nload positive file:{}".format(data_pos))
        p_choice = []
        fstr = make_str(data_pos)
        if os.path.exists("../save/randomdata/"+fstr + str(num) + ".txt"):
            with open("../save/randomdata/"+fstr + str(num) + ".txt", encoding="UTF-8", mode="r+") as f:
                for random_n in f:
                    random_n = random_n.replace("\n", "")
                    p_choice.append(int(random_n))
            f.close()
        else:
            p_choice = generate_random(num, data_pos)

        x_pos, tag_x_pos, p_pivots_pos, n_pivots_pos, pickup_pline, = load_tar(p_choice, f1, word_to_id, max_seq, FLAGS,
                                                                               transform=transform)
        y_pos = []
        for i in range(len(x_pos)):
            y_pos.append([1., 0.])

    with open(data_neg, encoding="UTF-8", mode="r+") as f2:
        print("\nload negative file:{}".format(data_neg))
        n_choice = []
        fstr = make_str(data_neg)
        if os.path.exists("../save/randomdata/"+fstr + str(num) + ".txt"):
            with open("../save/randomdata/"+fstr + str(num) + ".txt", encoding="UTF-8", mode="r+") as f:
                for random_n in f:
                    random_n = random_n.replace("\n", "")
                    n_choice.append(int(random_n))
            f.close()
        else:
            n_choice = generate_random(num, data_neg)
        x_neg, tag_x_neg, p_pivots_neg, n_pivots_neg, pickup_nline = load_tar(n_choice, f2, word_to_id, max_seq, FLAGS,
                                                                              transform=transform)
        y_neg = []
        for i in range(len(x_neg)):
            y_neg.append([0., 1.])

    if (shuffle):  # 一积一消
        x, y, tag_x, p_pivots, n_pivots = [], [], [], [], []
        pos_num = len(x_pos)
        neg_num = len(x_neg)
        m, n = 0, 0

        for i in range(pos_num + neg_num):
            if i % 2 == 0:
                x.append(x_pos[m])
                y.append(y_pos[m])
                tag_x.append(tag_x_pos[m])
                p_pivots.append(p_pivots_pos[m])
                n_pivots.append(n_pivots_pos[m])
                m += 1

            else:
                x.append(x_neg[n])
                y.append(y_neg[n])
                tag_x.append(tag_x_neg[n])
                p_pivots.append(p_pivots_neg[n])
                n_pivots.append(n_pivots_neg[n])
                n += 1

    else:  # 先积后消
        x, y, tag_x, p_pivots, n_pivots = [], [], [], [], []

        for i in range(len(x_pos)):
            x.append(x_pos[i])
            y.append(y_pos[i])
            tag_x.append(tag_x_pos[i])
            p_pivots.append(p_pivots_pos[i])
            n_pivots.append(n_pivots_pos[i])
        for i in range(len(x_neg)):
            x.append(x_neg[i])
            y.append(y_neg[i])
            tag_x.append(tag_x_neg[i])
            p_pivots.append(p_pivots_neg[i])
            n_pivots.append(n_pivots_neg[i])

    if transform:

        fw = open("../save/datafile/"+FLAGS.tar_domain+"_tar_trans.txt", 'wb')
        pickle.dump(np.asarray(x), fw, -1)  # Pickle dictionary using protocol 0.
        pickle.dump(np.asarray(y), fw)
        pickle.dump(np.asarray(tag_x), fw)
        pickle.dump(np.asarray(p_pivots), fw)
        pickle.dump(np.asarray(n_pivots), fw)

    else:
        fw = open("../save/datafile/"+FLAGS.tar_domain+"_tar.txt", 'wb')
        pickle.dump(np.asarray(x), fw, -1)  # Pickle dictionary using protocol 0.
        pickle.dump(np.asarray(y), fw)
        pickle.dump(np.asarray(tag_x), fw)


def load_test(choice, file, word_to_id, max_seq, FLAGS, type=None, encoding='UTF-8', transform=False):
    x, y, = [], [],
    pickup_line = []
    if transform:
        p_pivots_list = load_p_pivots_list(FLAGS.src_domain, FLAGS.tar_domain)
        n_pivots_list = load_n_pivots_list(FLAGS.src_domain, FLAGS.tar_domain)

    for findex, l1 in enumerate(file):
        if findex in choice:
            pickup_line.append(l1)
            t_x = np.zeros(max_seq)  # 存放数据，值为id,
            j = 0  # 计sentence里的word数
            sentence = cleanSentences(l1)
            sentence = sentence.strip()
            sentence = sentence.replace("\n", "")
            if sentence == "":
                pass
            else:
                words = sentence.split()
                if transform:
                    words, _, _ = transform_data(words, p_pivots_list, n_pivots_list)
                for word in words:
                    if j < max_seq:
                        if word in word_to_id:  # 单词在word_to_id里，
                            t_x[j] = word_to_id[word]

                        elif word == "UNK":
                            t_x[j] = len(word_to_id)
                        else:
                            print("词是：%s 不在word_to_id里" % word)
                        j += 1
                    else:  # 超过最大单词数截断
                        break

            x.append(t_x)

    print("the length of x:%d" % len(x))

    return np.asarray(x), pickup_line


def test_data(num, data_pos, data_neg, word_to_id, max_seq, FLAGS, shuffle=False, transform=False):
    with open(data_pos, encoding="UTF-8", mode="r+") as f1:
        print("\nload positive file:{}".format(data_pos))
        p_choice = []
        fstr = make_str(data_pos)
        if os.path.exists("../save/randomdata/"+fstr + str(num) + ".txt"):
            with open("../save/randomdata/"+fstr + str(num) + ".txt", encoding="UTF-8", mode="r+") as f:
                for random_n in f:
                    random_n = random_n.replace("\n", "")
                    p_choice.append(int(random_n))
            f.close()
        else:
            p_choice = generate_random(num, data_pos)

        x_pos, pickup_pline = load_test(p_choice, f1, word_to_id, max_seq, FLAGS, transform=transform)
        y_pos = []
        for i in range(len(x_pos)):
            y_pos.append([1., 0.])

    with open(data_neg, encoding="UTF-8", mode="r+") as f2:
        print("\nload negative file:{}".format(data_neg))
        n_choice = []
        fstr = make_str(data_neg)
        if os.path.exists("../save/randomdata/"+fstr + str(num) + ".txt"):
            with open("../save/randomdata/"+fstr + str(num) + ".txt", encoding="UTF-8", mode="r+") as f:
                for random_n in f:
                    random_n = random_n.replace("\n", "")
                    n_choice.append(int(random_n))
            f.close()
        else:
            n_choice = generate_random(num, data_neg)
        x_neg, pickup_nline = load_test(n_choice, f2, word_to_id, max_seq, FLAGS, transform=transform)
        y_neg = []
        for i in range(len(x_neg)):
            y_neg.append([0., 1.])

    if (shuffle):  # 一积一消
        x, y = [], []
        pos_num = len(x_pos)
        neg_num = len(x_neg)
        m, n = 0, 0

        for i in range(pos_num + neg_num):
            if i % 2 == 0:
                x.append(x_pos[m])
                y.append(y_pos[m])
                m += 1

            else:
                x.append(x_neg[n])
                y.append(y_neg[n])
                n += 1

        return np.asarray(x), np.asarray(y)
    else:  # 先积后消
        x, y, sen_len, doc_len = [], [], [], []

        for i in range(len(x_pos)):
            x.append(x_pos[i])
            y.append(y_pos[i])

        for i in range(len(x_neg)):
            x.append(x_neg[i])
            y.append(y_neg[i])

        return np.asarray(x), np.asarray(y)


def load_sdata(src_file_pos, src_file_neg, word_to_id, FLAGS):
    if not os.path.exists(r"../save/datafile/"+FLAGS.src_domain+"_src.txt"):
        src_data(src_file_pos, src_file_neg, word_to_id, FLAGS.max_seq, FLAGS, shuffle=False)

    fr_src = open("../save/datafile/"+FLAGS.src_domain+"_src.txt", 'rb')
    source_x = pickle.load(fr_src)
    source_doc_y = pickle.load(fr_src)
    source_tag_x = pickle.load(fr_src)

    return source_x, source_doc_y, source_tag_x


def load_tdata(tar_file_pos, tar_file_neg, word_to_id, FLAGS):
    if not os.path.exists(r"../save/datafile/"+FLAGS.tar_domain+"_tar.txt"):
        tar_data(FLAGS.domain_num, tar_file_pos, tar_file_neg, word_to_id, FLAGS.max_seq, FLAGS, shuffle=False)

    fr_tar = open("../save/datafile/"+FLAGS.tar_domain+"_tar.txt", 'rb')
    train_target_x = pickle.load(fr_tar)
    train_target_y = pickle.load(fr_tar)
    train_target_tag_x = pickle.load(fr_tar)

    return train_target_x, train_target_y, train_target_tag_x


def load_trans_sdata(src_file_pos, src_file_neg, word_to_id, FLAGS):
    if not os.path.exists(r"../save/datafile/"+FLAGS.src_domain+"_src_trans.txt"):
        src_data(src_file_pos, src_file_neg, word_to_id, FLAGS.max_seq, FLAGS, shuffle=False, transform=True)

    fr_src_trans = open("../save/datafile/"+FLAGS.src_domain+"_src_trans.txt", 'rb')
    source_x_trans = pickle.load(fr_src_trans)
    source_doc_y_trans = pickle.load(fr_src_trans)
    source_tag_x_trans = pickle.load(fr_src_trans)
    source_p_pivots_trans = pickle.load(fr_src_trans)
    source_n_pivots_trans = pickle.load(fr_src_trans)

    return source_x_trans, source_doc_y_trans, source_tag_x_trans, source_p_pivots_trans, source_n_pivots_trans


def load_trans_tdata(tar_file_pos, tar_file_neg, word_to_id, FLAGS):
    if not os.path.exists(r"../save/datafile/"+FLAGS.tar_domain+"_tar_trans.txt"):
        tar_data(FLAGS.domain_num, tar_file_pos, tar_file_neg, word_to_id, FLAGS.max_seq, FLAGS, shuffle=False,
                 transform=True)
    fr_tar_trans = open("../save/datafile/"+FLAGS.tar_domain+"_tar_trans.txt", 'rb')
    train_target_x_trans = pickle.load(fr_tar_trans)
    train_target_y_trans = pickle.load(fr_tar_trans)
    train_target_tag_x_trans = pickle.load(fr_tar_trans)
    train_target_p_pivots_trans = pickle.load(fr_tar_trans)
    train_target_n_pivots_trans = pickle.load(fr_tar_trans)

    return train_target_x_trans, train_target_y_trans, train_target_tag_x_trans, train_target_p_pivots_trans, train_target_n_pivots_trans
