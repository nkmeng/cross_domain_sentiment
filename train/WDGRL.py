# coding=UTF-8  

"""
# @File  : WDGRL.py
# @Author: HM
# @Date  : 2018/5/30
"""

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json

from train.config import FLAGS
from train.data_helper_WDGRL import *
from train.utils import *

word_to_id = load_word2id(FLAGS.word2id)

src_file_pos, src_file_neg, tar_file_pos, tar_file_neg = set_file(FLAGS.src_domain, FLAGS.tar_domain, FLAGS)


# RNN Model
class PModel(object):
    def __init__(self, wordVectors):
        self.embeddings = wordVectors
        self.is_train = tf.placeholder(tf.bool, [], name="train_flag")  # 标记是否在训练
        self.learning_rate1 = tf.placeholder(tf.float32, [], name="learnning_rate")

        self.lr_wd_D = tf.placeholder(tf.float32, [], name="learnning_rate_wd_D")

        self.X = tf.placeholder(tf.int32, [None, FLAGS.max_seq], name="input_data")  # 100*250 FLAGS.batch_size
        self.y_ = tf.placeholder(tf.float32, [None, FLAGS.n_classes], name="input_label")  # 100*2 FLAGS.batch_size
        self.tag_x = tf.placeholder(tf.int32, [None, FLAGS.max_seq], name="input_pos_tag")

        def weight_variable(shape):
            """Create a weight variable with appropriate initialization."""
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape, name):
            """Create a bias variable with appropriate initialization."""
            initial = tf.constant(0.1, shape=shape, name=name)
            return tf.Variable(initial)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.get_variable("embedding", shape=[FLAGS.vocab_size, FLAGS.word_dim],
                                initializer=tf.constant_initializer(self.embeddings),
                                trainable=True)
            inputs = tf.nn.embedding_lookup(W, self.X)

            inputs_expanded = tf.expand_dims(inputs, -1)  # 将inputs后边加一维，变成[batch_size,max_seq,word_dim,1]

        with tf.variable_scope('feature_extractor'):
            inputs = tf.nn.dropout(inputs, FLAGS.keep_prob)

            gru_cells = [tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size) for _ in range(FLAGS.n_layers)]
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(gru_cells)
            outputs, states = tf.nn.dynamic_rnn(stacked_lstm, inputs=inputs, dtype=tf.float32, time_major=False)

            output, self.alpha = attention(outputs, FLAGS.attention_dim, FLAGS.l2_reg_lambda)  # (?,300)
            self.feature = output

        with tf.variable_scope('sentiment_pred'):
            all_features = lambda: self.feature
            source_feature = lambda: tf.slice(self.feature, [0, 0], [FLAGS.batch_size // 2, -1])  # 从源域取一半的batch_size行
            target_feature = lambda: tf.slice(self.feature, [FLAGS.batch_size // 2, 0],
                                              [FLAGS.batch_size // 2, -1])  # 从源域取一半的batch_size行
            s_feats = tf.cond(self.is_train, source_feature, all_features)
            t_feats = tf.cond(self.is_train, target_feature, all_features)

            all_labels = lambda: self.y_
            source_labels = lambda: tf.slice(self.y_, [0, 0], [FLAGS.batch_size // 2, -1])
            target_labels = lambda: tf.slice(self.y_, [FLAGS.batch_size // 2, 0], [FLAGS.batch_size // 2, -1])
            self.s_labels = tf.cond(self.is_train, source_labels, all_labels)

            S_W = weight_variable([FLAGS.hidden_size, FLAGS.n_classes])
            S_b = bias_variable([FLAGS.n_classes], "S_bias")
            s_logits = tf.matmul(s_feats, S_W) + S_b  # classify_feats:[?,5000]

        alpha = tf.random_uniform(shape=[FLAGS.batch_size // 2, 1], minval=0., maxval=1.)
        differences = s_feats - t_feats
        interpolates = t_feats + (alpha * differences)
        h1_whole = tf.concat([self.feature, interpolates], 0)

        with tf.name_scope('critic'):
            C_W = weight_variable([FLAGS.hidden_size, 100])
            C_b = bias_variable([100], "C_bias")
            critic_h1 = tf.nn.relu(tf.matmul(h1_whole, C_W) + C_b)

            # = fc_layer(h1_whole, FLAGS.hidden_size, 100, layer_name='critic_h1')
            C_W_ = weight_variable([100, 1])
            C_b_ = bias_variable([1], "C_bias_")
            critic_out = tf.identity(tf.matmul(critic_h1, C_W_) + C_b_)
            # critic_out = fc_layer(critic_h1, 100, 1, layer_name='critic_h2', act=tf.identity)

        critic_s = tf.cond(self.is_train, lambda: tf.slice(critic_out, [0, 0], [FLAGS.batch_size // 2, -1]),
                           lambda: critic_out)
        critic_t = tf.cond(self.is_train,
                           lambda: tf.slice(critic_out, [FLAGS.batch_size // 2, 0], [FLAGS.batch_size // 2, -1]),
                           lambda: critic_out)

        with tf.name_scope("accuracy"):
            self.s_pred = tf.argmax(s_logits, 1)
            s_correct_prediction = tf.equal(self.s_pred, tf.argmax(self.s_labels, 1))
            self.s_correct_num = tf.reduce_sum(tf.cast(s_correct_prediction, tf.float32))
            self.s_acc = tf.reduce_mean(tf.cast(s_correct_prediction, tf.float32), name="sen_acc")

        with tf.name_scope("loss"):
            self.senti_loss = tf.nn.softmax_cross_entropy_with_logits(logits=s_logits, labels=self.s_labels)
            self.senti_cost = tf.reduce_mean(self.senti_loss)

            wd_loss = (tf.reduce_mean(critic_s) - tf.reduce_mean(critic_t))
            tf.summary.scalar('wd_loss', wd_loss)
            gradients = tf.gradients(critic_out, [h1_whole])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            tf.summary.scalar('gradient_penalty', gradient_penalty)
            theta_C = [v for v in tf.global_variables() if 'sentiment_pred' in v.name]
            theta_D = [v for v in tf.global_variables() if 'critic' in v.name]
            theta_G = [v for v in tf.global_variables() if 'feature_extractor1' in v.name]
            self.wd_d_op = tf.train.AdamOptimizer(self.lr_wd_D).minimize(-wd_loss + 10 * gradient_penalty,
                                                                         var_list=theta_D)
            all_variables = tf.trainable_variables()
            l2_loss = 1e-4 * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])
            total_loss = self.senti_cost + l2_loss + 1 * wd_loss
            self.train_op = tf.train.AdamOptimizer(self.learning_rate1).minimize(total_loss, var_list=theta_G + theta_C)

class Model(object):
    def __init__(self, wordVectors):
        self.embeddings = wordVectors
        self.domain = tf.placeholder(tf.float32, [None, 2], "domain_class")  # FLAGS.batch_size
        self.l = tf.placeholder(tf.float32, [], name="l")  #
        self.is_train = tf.placeholder(tf.bool, [], name="train_flag")  # 标记是否在训练
        self.learning_rate1 = tf.placeholder(tf.float32, [], name="learnning_rate")
        self.lr_wd_D = tf.placeholder(tf.float32, [], name="learnning_rate_wd_D")

        self.X = tf.placeholder(tf.int32, [None, FLAGS.max_seq], name="input_data")  # 100*250 FLAGS.batch_size
        self.y_ = tf.placeholder(tf.float32, [None, FLAGS.n_classes], name="input_label")  # 100*2 FLAGS.batch_size
        self.tag_x = tf.placeholder(tf.int32, [None, FLAGS.max_seq], name="input_pos_tag")

        # 转化后的
        self.is_test = tf.placeholder(tf.bool, [], name="train_flag")  # 标记是否在训练
        self.X_trans = tf.placeholder(tf.int32, [None, FLAGS.max_seq], name="input_data")  # 100*250 FLAGS.batch_size
        self.p_pivots_trans = tf.placeholder(tf.float32, [None, FLAGS.n_classes], name="pos_pivots_labels")
        self.n_pivots_trans = tf.placeholder(tf.float32, [None, FLAGS.n_classes], name="neg_pivots_labels")
        self.learning_rate3 = tf.placeholder(tf.float32, [], name="joint_learnning_rate")

        def weight_variable(shape):
            """Create a weight variable with appropriate initialization."""
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape, name):
            """Create a bias variable with appropriate initialization."""
            initial = tf.constant(0.1, shape=shape, name=name)
            return tf.Variable(initial)

        # with tf.device('/cpu:0'), tf.name_scope("embedding"):
        with tf.device('/cpu:0'), tf.name_scope("embedding1"):
            W = tf.get_variable("embedding1", shape=[FLAGS.vocab_size, FLAGS.word_dim],
                                initializer=tf.constant_initializer(self.embeddings),
                                trainable=True)
            inputs = tf.nn.embedding_lookup(W, self.X)

        with tf.device('/cpu:0'), tf.name_scope("embedding2"):
            W_trans = tf.get_variable("embedding2", shape=[FLAGS.vocab_size, FLAGS.word_dim],
                                      initializer=tf.constant_initializer(self.embeddings),
                                      trainable=True)
            inputs_trans = tf.nn.embedding_lookup(W_trans, self.X_trans)

        # bilstm model for feature extraction
        with tf.variable_scope('feature_extractor1'):
            inputs = tf.nn.dropout(inputs, FLAGS.keep_prob)
            # gru_cells = [tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size) for _ in range(FLAGS.n_layers)]
            # stacked_fw = tf.contrib.rnn.MultiRNNCell(gru_cells, state_is_tuple=True)
            # stacked_bw = tf.contrib.rnn.MultiRNNCell(gru_cells, state_is_tuple=True)
            # outputs, states = tf.nn.bidirectional_dynamic_rnn(stacked_fw, stacked_bw, inputs=inputs,
            #                                                   dtype=tf.float32)  # ,time_major=False
            # # outputs(?,200,150)
            # outputs = tf.concat(outputs, 2)  # (?,200,300)
            gru_cells = [tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size) for _ in range(FLAGS.n_layers)]
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(gru_cells)
            outputs, states = tf.nn.dynamic_rnn(stacked_lstm, inputs=inputs, dtype=tf.float32, time_major=False)

            output, self.alpha = attention(outputs, FLAGS.attention_dim, FLAGS.l2_reg_lambda)  # (?,300)
            # (?,300)
            # 域不变特征
            self.feature = output
            # self.feature = tf.reshape(output, [-1, FLAGS.hidden_size * FLAGS.batch_size])  # ??????

        with tf.variable_scope('feature_extractor_2'):
            inputs_trans = tf.nn.dropout(inputs_trans, FLAGS.keep_prob)
            gru_cells = [tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size) for _ in range(FLAGS.n_layers)]
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(gru_cells)
            outputs_trans, states_trans = tf.nn.dynamic_rnn(stacked_lstm, inputs=inputs_trans, dtype=tf.float32,
                                                            time_major=False)
            # gru_cells = [tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size) for _ in range(FLAGS.n_layers)]
            # stacked_fw = tf.contrib.rnn.MultiRNNCell(gru_cells, state_is_tuple=True)
            # stacked_bw = tf.contrib.rnn.MultiRNNCell(gru_cells, state_is_tuple=True)
            # outputs_trans, states = tf.nn.bidirectional_dynamic_rnn(stacked_fw, stacked_bw, inputs=inputs,
            #                                                         dtype=tf.float32)  # ,time_major=False
            # outputs_trans = tf.concat(outputs_trans, 2)
            output_trans, self.alpha_trans = attention_trans(outputs_trans, FLAGS.attention_dim,
                                                             FLAGS.l2_reg_lambda)  # (?,300)
            # 域不变特征
            self.feature_trans = output_trans
            # self.feature = tf.reshape(output, [-1, FLAGS.hidden_size * FLAGS.batch_size])  # ??????

        with tf.variable_scope('sentiment_pred'):
            # 如果is_train为true,classify_feats就为source_feature  即一半batch_size的数据
            # is_train为false,classify_feats为all_features 为batch_size的全部数据
            all_features = lambda: self.feature
            source_feature = lambda: tf.slice(self.feature, [0, 0], [FLAGS.batch_size // 2, -1])  # 从源域取一半的batch_size行
            target_feature = lambda: tf.slice(self.feature, [FLAGS.batch_size // 2, 0],
                                              [FLAGS.batch_size // 2, -1])  # 从源域取一半的batch_size行
            s_feats = tf.cond(self.is_train, source_feature, all_features)
            t_feats = tf.cond(self.is_train, target_feature, all_features)

            all_labels = lambda: self.y_
            source_labels = lambda: tf.slice(self.y_, [0, 0], [FLAGS.batch_size // 2, -1])
            self.s_labels = tf.cond(self.is_train, source_labels, all_labels)

            S_W = weight_variable([FLAGS.hidden_size, FLAGS.n_classes])
            S_b = bias_variable([FLAGS.n_classes], "S_bias")
            s_logits = tf.matmul(s_feats, S_W) + S_b  # classify_feats:[?,5000]

        alpha = tf.random_uniform(shape=[FLAGS.batch_size // 2, 1], minval=0., maxval=1.)
        differences = s_feats - t_feats
        interpolates = t_feats + (alpha * differences)
        h1_whole = tf.concat([self.feature, interpolates], 0)

        with tf.name_scope('critic'):
            C_W = weight_variable([FLAGS.hidden_size, 100])
            C_b = bias_variable([100], "C_bias")
            critic_h1 = tf.nn.relu(tf.matmul(h1_whole, C_W) + C_b)

            # = fc_layer(h1_whole, FLAGS.hidden_size, 100, layer_name='critic_h1')
            C_W_ = weight_variable([100, 1])
            C_b_ = bias_variable([1], "C_bias_")
            critic_out = tf.identity(tf.matmul(critic_h1, C_W_) + C_b_)
            # critic_out = fc_layer(critic_h1, 100, 1, layer_name='critic_h2', act=tf.identity)

        critic_s = tf.cond(self.is_train, lambda: tf.slice(critic_out, [0, 0], [FLAGS.batch_size // 2, -1]),
                           lambda: critic_out)
        critic_t = tf.cond(self.is_train,
                           lambda: tf.slice(critic_out, [FLAGS.batch_size // 2, 0], [FLAGS.batch_size // 2, -1]),
                           lambda: critic_out)

        # with tf.variable_scope('domain_pred'):
        #     # Flip the gradient when backpropagating through this operation
        #     feat = flip_gradient(self.feature, self.l)
        #
        #     d_W = weight_variable([FLAGS.hidden_size , 2])
        #     d_b = bias_variable([2])
        #     d_logits = tf.matmul(feat, d_W) + d_b

        with tf.variable_scope('pivots_pred_pos'):
            p_W_p = weight_variable([FLAGS.hidden_size, 2])
            p_b_p = bias_variable([2], "p_bias")
            p_logits_p = tf.matmul(self.feature_trans, p_W_p) + p_b_p

        with tf.variable_scope('pivots_pred_neg'):
            p_W_n = weight_variable([FLAGS.hidden_size, 2])
            p_b_n = bias_variable([2], "n_bias")
            p_logits_n = tf.matmul(self.feature_trans, p_W_n) + p_b_n

        with tf.variable_scope('final_pred'):
            outputs_half = lambda: tf.slice(self.feature, [0, 0],
                                            [FLAGS.batch_size // 2, -1])  # batch_size\2,200)
            out_doc = tf.cond(self.is_test, lambda: self.feature, outputs_half)

            outputs_trans_half = lambda: tf.slice(self.feature_trans, [0, 0],
                                                  [FLAGS.batch_size // 2, -1])  # (batch_size\2,200)

            out_doc_trans = tf.cond(self.is_test, lambda: self.feature_trans, outputs_trans_half)

            y_half = lambda: tf.slice(self.y_, [0, 0], [FLAGS.batch_size // 2, -1])
            self.j_y = tf.cond(self.is_test, lambda: self.y_, y_half)

            outputs = tf.concat([out_doc, out_doc_trans], 1)

            joint_W = weight_variable([FLAGS.hidden_size * 2, 2])
            joint_b = bias_variable([2], "joint_bias")
            j_logits = tf.matmul(outputs, joint_W) + joint_b

        with tf.name_scope("loss"):
            self.senti_loss = tf.nn.softmax_cross_entropy_with_logits(logits=s_logits, labels=self.s_labels)
            # self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=self.domain)
            self.p_pivots_loss = tf.nn.softmax_cross_entropy_with_logits(logits=p_logits_p, labels=self.p_pivots_trans)
            self.n_pivots_loss = tf.nn.softmax_cross_entropy_with_logits(logits=p_logits_n, labels=self.n_pivots_trans)
            self.joint_loss = tf.nn.softmax_cross_entropy_with_logits(logits=j_logits, labels=self.j_y)

            self.senti_cost = tf.reduce_mean(self.senti_loss)
            # self.domain_cost = tf.reduce_mean(self.domain_loss)
            self.p_pivots_cost = tf.reduce_mean(self.p_pivots_loss)
            self.n_pivots_cost = tf.reduce_mean(self.n_pivots_loss)
            self.joint_cost = tf.reduce_mean(self.joint_loss)

            wd_loss = (tf.reduce_mean(critic_s) - tf.reduce_mean(critic_t))
            gradients = tf.gradients(critic_out, [h1_whole])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            theta_C = [v for v in tf.global_variables() if 'sentiment_pred' in v.name]
            theta_D = [v for v in tf.global_variables() if 'critic' in v.name]
            theta_G = [v for v in tf.global_variables() if 'feature_extractor1' in v.name]
            self.wd_d_op = tf.train.AdamOptimizer(self.lr_wd_D).minimize(-wd_loss + 10 * gradient_penalty,
                                                                         var_list=theta_D)
            all_variables = tf.trainable_variables()
            train_varaiables = 0
            # l2_loss = 1e-4 * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, C_W)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, C_W_)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, p_W_p)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, p_W_n)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, joint_W)
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.005)
            reg_term = tf.contrib.layers.apply_regularization(regularizer)
            self.total_loss = self.joint_cost + self.p_pivots_cost + self.n_pivots_cost + 1 * wd_loss + reg_term  # + l2_loss
            self.train_op = tf.train.AdamOptimizer(self.learning_rate1).minimize(self.total_loss,
                                                                                 var_list=theta_G + theta_C)

            # # tf.add_to_collection(tf.GraphKeys.WEIGHTS, S_W)
            # tf.add_to_collection(tf.GraphKeys.WEIGHTS, d_W)
            # tf.add_to_collection(tf.GraphKeys.WEIGHTS, p_W_p)
            # tf.add_to_collection(tf.GraphKeys.WEIGHTS, p_W_n)
            # tf.add_to_collection(tf.GraphKeys.WEIGHTS, joint_W)
            # regularizer = tf.contrib.layers.l2_regularizer(scale=0.005)
            # reg_term = tf.contrib.layers.apply_regularization(regularizer)
            # self.total_loss = self.senti_cost + self.domain_cost + reg_term
            #
            # self.jloss = self.joint_cost + self.domain_cost + self.p_pivots_cost + self.n_pivots_cost  # + self.senti_cost
            # self.final_loss = self.jloss
            # self.final_loss += reg_term

        with tf.name_scope("accuracy"):
            self.s_pred = tf.argmax(s_logits, 1)
            s_correct_prediction = tf.equal(self.s_pred, tf.argmax(self.s_labels, 1))
            self.s_correct_num = tf.reduce_sum(tf.cast(s_correct_prediction, tf.float32))
            self.s_acc = tf.reduce_mean(tf.cast(s_correct_prediction, tf.float32), name="sen_acc")
            #
            # self.domain_pred = tf.argmax(d_logits, 1)
            # d_correct_prediction = tf.equal(self.domain_pred, tf.argmax(self.domain, 1))
            # self.d_correct_num = tf.reduce_sum(tf.cast(d_correct_prediction, tf.float32))
            # self.d_acc = tf.reduce_mean(tf.cast(d_correct_prediction, tf.float32), name="domian_acc")

            self.p_pivots_pred = tf.argmax(p_logits_p, 1)
            p_p_correct_prediction = tf.equal(self.p_pivots_pred, tf.argmax(self.p_pivots_trans, 1))
            self.p_p_correct_num = tf.reduce_sum(tf.cast(p_p_correct_prediction, tf.float32))
            self.p_p_acc = tf.reduce_mean(tf.cast(p_p_correct_prediction, tf.float32), name="p_pivots_accuracy")

            self.n_pivots_pred = tf.argmax(p_logits_n, 1)
            p_n_correct_prediction = tf.equal(self.n_pivots_pred, tf.argmax(self.n_pivots_trans, 1))
            self.p_n_correct_num = tf.reduce_sum(tf.cast(p_n_correct_prediction, tf.float32))
            self.p_n_acc = tf.reduce_mean(tf.cast(p_n_correct_prediction, tf.float32), name="n_pivots_accuracy")

            self.joint_pred = tf.argmax(j_logits, 1)
            j_correct_prediction = tf.equal(self.joint_pred, tf.argmax(self.j_y, 1))
            self.j_correct_num = tf.reduce_sum(tf.cast(j_correct_prediction, tf.float32))
            self.j_acc = tf.reduce_mean(tf.cast(j_correct_prediction, tf.float32), name="joint_train_accuracy")

            # self.dann_train_op = tf.train.AdamOptimizer(self.learning_rate1).minimize(self.total_loss)
            # self.dann_train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.total_loss)
            # self.pivots_train_op = tf.train.MomentumOptimizer(self.learning_rate2, 0.9).minimize(self.pivots_cost)
            # self.joint_train_op = tf.train.AdamOptimizer(self.learning_rate3).minimize(self.final_loss)
            # self.joint_train_op = tf.train.MomentumOptimizer(self.learning_rate3, 0.9).minimize(self.final_loss)
            # self.joint_train_op = tf.train.GradientDescentOptimizer(self.learning_rate3).minimize(self.final_loss)

            #
            # # add summary
            # sen_loss_summary = tf.summary.scalar("sentiment_loss", self.senti_cost)
            # dom_loss_summary = tf.summary.scalar("domian_loss", self.domain_cost)
            # # add summary
            # sen_acc_summary = tf.summary.scalar("sentiment_accuracy", self.s_acc)
            # dom_acc_summary = tf.summary.scalar("domain_accuracy", self.d_acc)
            #
            # # self.summary = tf.summary.merge([sen_loss_summary, dom_loss_summary, sen_acc_summary, dom_acc_summary])
            # self.summary = tf.summary.merge_all()
            # self.test_summary = tf.summary.merge([sen_loss_summary, sen_acc_summary])


def train_dann(model, verbose=False):
    saver = tf.train.Saver(max_to_keep=2)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # model_file = tf.train.latest_checkpoint("../save/model/"+FLAGS.src_domain + "_" + FLAGS.tar_domain + 'P_net/')
        # saver.restore(sess, model_file)

        source_x, source_doc_y, source_tag_x = load_sdata(src_file_pos, src_file_neg, word_to_id, FLAGS)
        train_target_x, train_target_y, train_target_tag_x = load_tdata(tar_file_pos, tar_file_neg, word_to_id, FLAGS)
        test_x, test_y = test_data(FLAGS.test_num, tar_file_pos, tar_file_neg, word_to_id, FLAGS.max_seq, FLAGS,
                                   shuffle=False)
        # train_x, valid_x, train_y, valid_y, train_tag_x, valid_tag_x = train_test_split(
        #     source_x, source_doc_y, source_tag_x, test_size=0.2, random_state=1)
        # random_state随机数种子：在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
        train_x, valid_x, train_y, valid_y, train_tag_x, valid_tag_x, = trainTestSplit(
            source_x, source_doc_y, source_tag_x, test_size=0.2, )


        best_accuracy = 0
        best_val_epoch = 0
        best_test_acc = 0
        best_test_accuracy = 0

        for epoch in range(FLAGS.epoches):

            i = 0
            # Adaptation param and learning rate schedule as described in the paper
            p = float(epoch) / FLAGS.epoches
            lr = 1e-3
            lr_wd_D = 1e-3
            print("=" * 20 + "Epoch", epoch, "=" * 20 + "\n")

            data_len = len(train_x)
            iterations = data_len // (FLAGS.batch_size // 2)
            for step in range(iterations):
                i += 1

                x0, y0, tag_x0 = next(
                    batches_iter(train_x, train_y, train_tag_x, batch_size=FLAGS.batch_size // 2))
                x1, y1, tag_x1 = next(
                    batches_iter(train_target_x, train_target_y, train_target_tag_x,
                                 batch_size=FLAGS.batch_size // 2))

                X = np.vstack([x0, x1])
                y = np.vstack([y0, y1])

                for _ in range(5):
                    _ = sess.run(model.wd_d_op, feed_dict={model.X: X, model.is_train: True,
                                                           model.lr_wd_D: lr_wd_D})

                _ = sess.run(model.train_op,
                             feed_dict={model.X: X, model.y_: y, model.is_train: True, model.learning_rate1: lr})

            acc_xs, c_loss_xs = sess.run([model.s_acc, model.senti_cost],
                                         feed_dict={model.X: valid_x, model.y_: valid_y,
                                                    model.is_train: False})

            acc_xt, c_loss_xt = sess.run([model.s_acc, model.senti_cost],
                                         feed_dict={model.X: test_x, model.y_: test_y,
                                                    model.is_train: False})

            print('epoch: ', epoch)
            print('Source classifier loss: %f, Target classifier loss: %f' % (c_loss_xs, c_loss_xt))
            print('Source label accuracy: %f, Target label accuracy: %f' % (acc_xs, acc_xt))
            if acc_xs > best_accuracy:
                best_accuracy = acc_xs
                best_val_epoch = epoch
                best_test_acc = acc_xt

                path = saver.save(sess, "../save/model/"+FLAGS.src_domain + "_" + FLAGS.tar_domain + 'P_net/model_0', epoch)
                print("Saved model checkpoint to {}\n".format(path))
                if best_test_accuracy < acc_xt:
                    best_test_accuracy = acc_xt


def load_stopword():
    stoplist = []
    with open("../stopword.txt", encoding="UTF-8", mode="r+") as f:
        for line in f:
            line = line.replace("\n", "")
            stoplist.append(line)
    f.close()
    return stoplist


def lookup_pivots(docs, tag_x):
    """
    找出由小到大排序的index
    找最大attention的word，找到最大的之后，还需要判断词性位置是否为1
    不为1就选下一个大的
    """
    flag = False
    p_num = 0
    max_word_index = []
    length = len(docs)  # 一条评论词的长度
    index_list = np.argsort(docs)[:]
    for i in range(length):
        if tag_x[index_list[-i - 1]] == 1:  # 是需要词性的attention最大的
            p_num += 1
            flag = True
            max_word_index.append(index_list[-i - 1])
        if p_num == 2:
            break
    if flag:  # 如果有需要词性的就返回
        return max_word_index
    else:
        return -1


def make_p_pivots(model):
    print("pick up pos pivots start:\n")

    saver = tf.train.Saver(max_to_keep=2)

    pivots = []
    pivots_id = []
    pivots_num = dict()

    with tf.Session() as sess:
        model_file = tf.train.latest_checkpoint("WDGRL/" + FLAGS.src_domain + "_" + FLAGS.tar_domain + 'P_net/')
        # model_file = tf.train.latest_checkpoint('train_P_net/')
        saver.restore(sess, model_file)

        id_to_word = load_id2word(FLAGS.word2id)

        source_x, source_doc_y, source_tag_x = load_sdata(src_file_pos, src_file_neg, word_to_id, FLAGS)
        train_target_x, train_target_y, train_target_tag_x = load_tdata(tar_file_pos, tar_file_neg, word_to_id, FLAGS)

        train_x, valid_x, train_y, valid_y, train_tag_x, valid_tag_x = trainTestSplit(
            source_x, source_doc_y, source_tag_x, test_size=0.2, )
        # train_x train_target_x 前一半都是积极，后一半都是消极？
        stoplist = load_stopword()
        global_step = 0
        data_len = len(train_x)

        train_x_pos = train_x[0:data_len // 2]
        train_y_pos = train_y[0:data_len // 2]
        train_tag_x_pos = train_tag_x[0:data_len // 2]
        train_target_x_pos = train_target_x[0:data_len // 2]
        train_target_y_pos = train_target_y[0:data_len // 2]
        train_target_tag_x_pos = train_target_tag_x[0:data_len // 2]

        iterations = (data_len // 2) // (FLAGS.batch_size // 2)
        for step in range(iterations):
            print("%d iterations:" % step)
            global_step += 1
            # for step, indices in enumerate(batch_index(len(train_y), model.config.batch_size//2, 1), 1):
            x0, y0, tag_x0 = next(
                batches_iter(train_x_pos, train_y_pos, train_tag_x_pos, batch_size=FLAGS.batch_size // 2, shuffle=True))
            x1, y1, tag_x1 = next(
                batches_iter(train_target_x_pos, train_target_y_pos, train_target_tag_x_pos,
                             batch_size=FLAGS.batch_size // 2,
                             shuffle=True))

            X = np.vstack([x0, x1])
            y = np.vstack([y0, y1])
            tag_x = np.vstack([tag_x0, tag_x1])

            alpha = sess.run(model.alpha, feed_dict={model.X: X, model.y_: y, model.tag_x: tag_x, model.is_train: True})

            pivots_index = []
            for d_i, docs in enumerate(alpha):
                pivots_index.append(lookup_pivots(docs, tag_x[d_i]))

            for X_i, X_data in enumerate(X):  # 对X的doucment级
                if pivots_index[X_i] != -1:  # 说明有需要词性的pivots
                    for p_i in pivots_index[X_i]:
                        if len(id_to_word[X_data[p_i]]) > 2:
                            print(id_to_word[X_data[p_i]])
                            if id_to_word[X_data[p_i]] not in stoplist:
                                if id_to_word[X_data[p_i]] not in pivots:
                                    pivots_id.append(X_data[p_i])
                                    pivots.append(id_to_word[X_data[p_i]])

            for X_i, X_data in enumerate(X):  # 对X的doucment级
                for w_i, w_data in enumerate(X_data):
                    if w_data in pivots_id:
                        if id_to_word[w_data] not in pivots_num.keys():
                            pivots_num[id_to_word[w_data]] = 1
                        else:
                            pivots_num[id_to_word[w_data]] += 1

            print(pivots_num)
            f = open('WDGRL/pivots/pivots_num_pos.txt', 'w')
            f.write(json.dumps(pivots_num))
            f.close()

            with open("WDGRL/pivots/" + FLAGS.src_domain + "_" + FLAGS.tar_domain + "_allword_pos.txt", mode="w+",
                      encoding="UTF-8") as f:
                for word in pivots:
                    f.write(word + '\n')
            f.close()


def make_n_pivots(model):
    print("pick up neg pivots start:\n")

    saver = tf.train.Saver(max_to_keep=2)

    pivots = []
    pivots_id = []
    pivots_num = dict()

    with tf.Session() as sess:
        # model_file = tf.train.latest_checkpoint('train_P_net/')
        model_file = tf.train.latest_checkpoint("WDGRL/" + FLAGS.src_domain + "_" + FLAGS.tar_domain + 'P_net/')
        saver.restore(sess, model_file)

        id_to_word = load_id2word(FLAGS.word2id)

        source_x, source_doc_y, source_tag_x = load_sdata(src_file_pos, src_file_neg, word_to_id, FLAGS)
        train_target_x, train_target_y, train_target_tag_x = load_tdata(tar_file_pos, tar_file_neg, word_to_id, FLAGS)

        train_x, valid_x, train_y, valid_y, train_tag_x, valid_tag_x = trainTestSplit(
            source_x, source_doc_y, source_tag_x, test_size=0.2, )

        stoplist = load_stopword()
        global_step = 0
        data_len = len(train_x)

        train_x_neg = train_x[data_len // 2:]
        train_y_neg = train_y[data_len // 2:]
        train_tag_x_neg = train_tag_x[data_len // 2:]
        train_target_x_neg = train_target_x[data_len // 2:]
        train_target_y_neg = train_target_y[data_len // 2:]
        train_target_tag_x_neg = train_target_tag_x[data_len // 2:]

        iterations = (data_len // 2) // (FLAGS.batch_size // 2)
        for step in range(iterations):
            print("%d iterations:" % step)
            global_step += 1
            # for step, indices in enumerate(batch_index(len(train_y), model.config.batch_size//2, 1), 1):
            x0, y0, tag_x0 = next(
                batches_iter(train_x_neg, train_y_neg, train_tag_x_neg, batch_size=FLAGS.batch_size // 2, shuffle=True))
            x1, y1, tag_x1 = next(
                batches_iter(train_target_x_neg, train_target_y_neg, train_target_tag_x_neg,
                             batch_size=FLAGS.batch_size // 2,
                             shuffle=True))

            X = np.vstack([x0, x1])
            y = np.vstack([y0, y1])
            tag_x = np.vstack([tag_x0, tag_x1])

            alpha = sess.run(model.alpha, feed_dict={model.X: X, model.y_: y, model.tag_x: tag_x, model.is_train: True})

            pivots_index = []
            for d_i, docs in enumerate(alpha):
                pivots_index.append(lookup_pivots(docs, tag_x[d_i]))

            for X_i, X_data in enumerate(X):  # 对X的doucment级
                if pivots_index[X_i] != -1:  # 说明有需要词性的pivots
                    for p_i in pivots_index[X_i]:
                        if len(id_to_word[X_data[p_i]]) > 2:
                            print(id_to_word[X_data[p_i]])
                            if id_to_word[X_data[p_i]] not in stoplist:
                                if id_to_word[X_data[p_i]] not in pivots:
                                    pivots_id.append(X_data[p_i])
                                    pivots.append(id_to_word[X_data[p_i]])

            for X_i, X_data in enumerate(X):  # 对X的doucment级
                for w_i, w_data in enumerate(X_data):
                    if w_data in pivots_id:
                        if id_to_word[w_data] not in pivots_num.keys():
                            pivots_num[id_to_word[w_data]] = 1
                        else:
                            pivots_num[id_to_word[w_data]] += 1

            print(pivots_num)
            f = open('WDGRL/pivots/pivots_num_neg.txt', 'w')
            f.write(json.dumps(pivots_num))
            f.close()

            with open("WDGRL/pivots/" + FLAGS.src_domain + "_" + FLAGS.tar_domain + "_allword_neg.txt", mode="w+",
                      encoding="UTF-8") as f:
                for word in pivots:
                    f.write(word + '\n')
            f.close()


def pickup_p_pivots():
    pivots = []
    f = open('WDGRL/pivots/pivots_num_pos.txt', 'r')
    pivots_num = json.loads(f.read())
    f.close()
    for word in pivots_num.keys():
        if pivots_num[word] >= 5:
            pivots.append(word)

    with open("WDGRL/pivots/" + FLAGS.src_domain + "_" + FLAGS.tar_domain + "_pos1.txt", mode="w+",
              encoding="UTF-8") as f:
        for word in pivots:
            f.write(word + '\n')
    f.close()


def pickup_n_pivots():
    pivots = []
    f = open('WDGRL/pivots/pivots_num_neg.txt', 'r')
    pivots_num = json.loads(f.read())
    f.close()
    for word in pivots_num.keys():
        if pivots_num[word] >= 5:
            pivots.append(word)

    with open("WDGRL/pivots/" + FLAGS.src_domain + "_" + FLAGS.tar_domain + "_neg1.txt", mode="w+",
              encoding="UTF-8") as f:
        for word in pivots:
            f.write(word + '\n')
    f.close()


def joint_train(model):
    print("Training start:\n")
    saver = tf.train.Saver(max_to_keep=3)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model_file = tf.train.latest_checkpoint("WDGRL/" + FLAGS.src_domain + "_" + FLAGS.tar_domain + 'joint/')
        saver.restore(sess, model_file)

        # model_file = tf.train.latest_checkpoint('joint_train/')
        # saver.restore(sess, model_file)

        source_x, source_doc_y, source_tag_x = load_sdata(src_file_pos, src_file_neg, word_to_id, FLAGS)
        train_target_x, train_target_y, train_target_tag_x = load_tdata(tar_file_pos, tar_file_neg, word_to_id, FLAGS)

        test_x, test_y = test_data(FLAGS.test_num, tar_file_pos, tar_file_neg, word_to_id, FLAGS.max_seq, FLAGS,
                                   shuffle=False)

        source_x_trans, source_doc_y_trans, source_tag_x_trans, source_p_pivots_trans, source_n_pivots_trans = load_trans_sdata(
            src_file_pos, src_file_neg, word_to_id, FLAGS)
        train_target_x_trans, train_target_y_trans, train_target_tag_x_trans, train_target_p_pivots_trans, train_target_n_pivots_trans = load_trans_tdata(
            tar_file_pos, tar_file_neg, word_to_id, FLAGS)

        train_x, valid_x, train_y, valid_y, train_tag_x, valid_tag_x, train_x_trans, valid_x_trans, train_y_trans, valid_y_trans, train_p_pivots_trans, valid_p_pivots_trans, train_n_pivots_trans, valid_n_pivots_trans = joint_trainTestSplit(
            source_x, source_doc_y, source_tag_x, source_x_trans, source_doc_y_trans, source_p_pivots_trans,
            source_n_pivots_trans, test_size=0.2)

        # 原始数据和转化后的数据不能打乱 因为两个网络要对应标签也要对应
        # 在batch_iter里打乱，但是源和转换的要对应
        test_x_trans, test_y_trans = test_data(FLAGS.test_num, tar_file_pos, tar_file_neg, word_to_id, FLAGS.max_seq,
                                               FLAGS, shuffle=False, transform=True)

        best_accuracy = 0
        best_val_epoch = 0
        best_test_accuracy = 0
        best_test_acc = 0

        domain_labels = np.vstack([np.tile([1., 0.], [FLAGS.batch_size // 2, 1]),
                                   np.tile([0., 1.], [FLAGS.batch_size // 2, 1])])

        for epoch in range(FLAGS.epoches):

            p = float(epoch) / FLAGS.epoches
            l = 2. / (1. + np.exp(-10. * p)) - 1
            l = min(l, 0.1)
            # lr3 = 0.075 / (1. + 10 * p) ** 0.75
            lr3 = 1e-6
            lr_wd_D = 1e-6

            print("=" * 20 + "Epoch", epoch, "=" * 20 + "\n")

            data_len = len(train_x)
            iterations = data_len // (FLAGS.batch_size // 2)

            # dann 训练
            i = 0
            for step in range(iterations):
                i += 1

                x0, y0, tag_x0, x0_trans, y0_trans, p_pivots0_trans, n_pivots0_trans = next(
                    joint_batches_iter(train_x, train_y, train_tag_x, train_x_trans, train_y_trans,
                                       train_p_pivots_trans, train_n_pivots_trans, batch_size=FLAGS.batch_size // 2))
                x1, y1, tag_x1, x1_trans, y1_trans, p_pivots1_trans, n_pivots1_trans = next(
                    joint_batches_iter(train_target_x, train_target_y, train_target_tag_x, train_target_x_trans,
                                       train_target_y_trans, train_target_p_pivots_trans, train_target_n_pivots_trans,
                                       batch_size=FLAGS.batch_size // 2))

                X = np.vstack([x0, x1])
                X_trans = np.vstack([x0_trans, x1_trans])

                y = np.vstack([y0, y1])
                p_pivots_trans = np.vstack([p_pivots0_trans, p_pivots1_trans])
                n_pivots_trans = np.vstack([n_pivots0_trans, n_pivots1_trans])
                tag_x = np.vstack([tag_x0, tag_x1])

                for _ in range(5):
                    _ = sess.run(model.wd_d_op, feed_dict={model.X: X, model.is_train: True,
                                                           model.lr_wd_D: lr_wd_D})
                    #
                    # feed_dict = {}  # model.learning_rate1: lr1, model.learning_rate2: lr2,model.learning_rate3: lr3

                _, final_loss, joint_cost, j_acc, pos_acc, neg_acc = sess.run(
                    [model.train_op, model.total_loss, model.joint_cost, model.j_acc, model.p_p_acc,
                     model.p_n_acc],
                    feed_dict={model.X: X, model.y_: y, model.tag_x: tag_x, model.domain: domain_labels,
                               model.is_test: False, model.is_train: True,
                               model.X_trans: X_trans, model.p_pivots_trans: p_pivots_trans,
                               model.n_pivots_trans: n_pivots_trans, model.l: l,
                               model.learning_rate1: lr3})
                print(
                    "Epoch %i: step %i: final_loss is:%f,j_loss is:%f,pos_acc is:%f,neg_acc is:%f, j_acc is:%f" % (
                        epoch, i, final_loss, joint_cost, pos_acc, neg_acc, j_acc))

            acc_xs, c_loss_xs = sess.run([model.j_acc, model.joint_cost],
                                         feed_dict={model.X: valid_x, model.X_trans: valid_x_trans, model.y_: valid_y,
                                                    model.is_train: False, model.is_test: True})

            test_x, test_y, test_x_trans = shuffle_test(test_x, test_y, test_x_trans)
            acc_xt, c_loss_xt = sess.run([model.j_acc, model.joint_cost],
                                         feed_dict={model.X: test_x, model.X_trans: test_x_trans, model.y_: test_y,
                                                    model.is_train: False, model.is_test: True})

            print('step: ', epoch)
            print('Source classifier loss: %f, Target classifier loss: %f' % (c_loss_xs, c_loss_xt))
            print('Source label accuracy: %f, Target label accuracy: %f' % (acc_xs, acc_xt))
            if acc_xs > best_accuracy:
                best_accuracy = acc_xs
                best_val_epoch = epoch
                best_test_acc = acc_xt

                path = saver.save(sess, "WDGRL/" + FLAGS.src_domain + "_" + FLAGS.tar_domain + 'joint/model_4', epoch)

                # path = saver.save(sess, 'joint_train/model_1', epoch)
                print("Saved model checkpoint to {}\n".format(path))
                if best_test_accuracy < acc_xt:
                    best_test_accuracy = acc_xt


with tf.Graph().as_default():
    wordVectors = np.load(FLAGS.wordVector_file)
    print('Loaded the word vectors!')
    print(wordVectors.shape)  # 输出：（54174，300）
    #
    model = PModel(wordVectors)
    print("\nstart training")

    # 1
    train_dann(model)

    # # # 2
    # make_p_pivots(model)

    # # 3
    # make_n_pivots(model)

    # 4
    # pickup_p_pivots()
    # pickup_n_pivots()

    # # # 5
    # mymodel = Model(wordVectors)
    # joint_train(mymodel)
