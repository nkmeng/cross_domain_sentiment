# coding=UTF-8  

"""
# @File  : utils.py
# @Author: HM
# @Date  : 2018/4/3
"""
import tensorflow as tf


def attention(inputs, attention_size, l2_reg_lambda):
    """
    Attention mechanism layer.
    :param inputs: outputs of RNN/Bi-RNN layer (not final state)
    :param attention_size: linear size of attention weights
    :return: outputs of the passed RNN/Bi-RNN reduced with attention vector
    """
    # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
    if isinstance(inputs, tuple):  # 判断inputs是否为tuple类型
        inputs = tf.concat(2, inputs)  # 将inputs在第2维上连接

    sequence_length = inputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs.get_shape()[2].value  # hidden size of the RNN layer

    # Attention mechanism
    W_omega = tf.get_variable("W_omega", initializer=tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.get_variable("b_omega", initializer=tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.get_variable("u_omega", initializer=tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    h_flag = tf.reshape(alphas, [-1, sequence_length, 1])

    # Output of Bi-RNN is reduced with attention vector
    output = tf.reduce_sum(inputs * h_flag, 1)
    # if l2_reg_lambda > 0:
    #    l2_loss += tf.nn.l2_loss(W_omega)
    #    l2_loss += tf.nn.l2_loss(b_omega)
    #    l2_loss += tf.nn.l2_loss(u_omega)
    #    tf.add_to_collection('losses', l2_loss)

    return output,alphas


def attention_trans(inputs, attention_size, l2_reg_lambda):
    """
    Attention mechanism layer.
    :param inputs: outputs of RNN/Bi-RNN layer (not final state)
    :param attention_size: linear size of attention weights
    :return: outputs of the passed RNN/Bi-RNN reduced with attention vector
    """
    # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
    if isinstance(inputs, tuple):  # 判断inputs是否为tuple类型
        inputs = tf.concat(2, inputs)  # 将inputs在第2维上连接

    sequence_length = inputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs.get_shape()[2].value  # hidden size of the RNN layer

    # Attention mechanism
    W_omega = tf.get_variable("W_omega", initializer=tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.get_variable("b_omega", initializer=tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.get_variable("u_omega", initializer=tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    h_flag = tf.reshape(alphas, [-1, sequence_length, 1])

    # Output of Bi-RNN is reduced with attention vector
    output = tf.reduce_sum(inputs * h_flag, 1)


    return output,alphas

