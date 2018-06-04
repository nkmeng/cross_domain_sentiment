# coding=UTF-8  

"""
# @File  : config.py
# @Author: HM
# @Date  : 2018/4/26
"""

import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string("summary_dir", "D://cross_domain/DANN_bilstm/B_D", "save tensorboard_graph")

flags.DEFINE_string("src_domain", "books", "books,dvd,ele,kit")
flags.DEFINE_string("tar_domain", "dvd", "books,dvd,ele,kit")

flags.DEFINE_integer("epoches", 50, "epoches")  # 120
flags.DEFINE_integer("batch_size", 64, "batch size of each batch")
flags.DEFINE_integer("hidden_size", 150, "hidden size rnn size of lstm")
flags.DEFINE_integer("n_layers", 1, "the num of rnn layer")
flags.DEFINE_integer("attention_dim", 200, "embedding size")
flags.DEFINE_float("keep_prob", 1, "keep_prob")
flags.DEFINE_float("early_stop", 10, "early_stopping ")

flags.DEFINE_integer("max_seq", 200, "Maximum length of sentence")
flags.DEFINE_integer("vocab_size", 118926, "the size of wordlist")  # 117040
flags.DEFINE_integer("word_dim", 300, "wordvec dimension")
flags.DEFINE_integer("test_num", 200, "test pos num = test neg num=200")
flags.DEFINE_integer("domain_num", 800, "test pos num = test neg num=800")
flags.DEFINE_integer("source_num", 1000, " pos num =  neg num=1000")

flags.DEFINE_integer("n_classes", 2, "the num of classes")

flags.DEFINE_integer("checkpoint_every", 110, "run evaluation")
flags.DEFINE_integer("evaluate_every", 80, "run evaluation")
flags.DEFINE_float("l2_reg_lambda", 0.05, "l2 regulation")
flags.DEFINE_string("word2id", "../word2vec/allword2id.txt", "word-id dict")

flags.DEFINE_string("wordList_file", "../word2vec/all_word_list.txt", "words python list")
flags.DEFINE_string("wordVector_file", "../word2vec/all_embedding_matrix.npy", "google embedding matrix")

# 训练源域
flags.DEFINE_string("books_pos", "../data/books/positive_processed", "data")
flags.DEFINE_string("books_neg", "../data/books/negative_processed", "data")
flags.DEFINE_string("dvd_pos", "../data/dvd/positive_processed", "data")
flags.DEFINE_string("dvd_neg", "../data/dvd/negative_processed", "data")
flags.DEFINE_string("ele_pos", "../data/electronics/positive_processed", "data")
flags.DEFINE_string("ele_neg", "../data/electronics/negative_processed", "data")
flags.DEFINE_string("kit_pos", "../data/kitchen_&_housewares/positive_processed", "data")
flags.DEFINE_string("kit_neg", "../data/kitchen_&_housewares/negative_processed", "data")

# 测试目标域
flags.DEFINE_string("books_u_neg", "../data/books/unlabeled_negative", "")
flags.DEFINE_string("books_u_pos", "../data/books/unlabeled_positive", "")
flags.DEFINE_string("dvd_u_neg", "../data/dvd/unlabeled_negative", "")
flags.DEFINE_string("dvd_u_pos", "../data/dvd/unlabeled_positive", "")
flags.DEFINE_string("ele_u_neg", "../data/electronics/unlabeled_negative", "")
flags.DEFINE_string("ele_u_pos", "../data/electronics/unlabeled_positive", "")
flags.DEFINE_string("kit_u_neg", "../data/kitchen_&_housewares/unlabeled_negative", "")
flags.DEFINE_string("kit_u_pos", "../data/kitchen_&_housewares/unlabeled_positive", "")

FLAGS = flags.FLAGS
