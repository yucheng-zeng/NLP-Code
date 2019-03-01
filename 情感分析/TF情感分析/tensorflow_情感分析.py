import os
import numpy as np
import tensorflow as tf
from random import randint

import warnings
warnings.filterwarnings('ignore', '.*do not.*',)

# tensorflow中gpu参数的一些设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 设置为3，用来屏蔽掉numpy的warning
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allocator_type = 'BFC'

# LSTM的部分参数设置
max_seq_num = 250
num_dimensions = 300
batch_size = 24
lstm_units = 64
num_labels = 2
iterations = 100000
ids = np.load('./data/idsMatrix.npy')
word_vectors = np.load('./data/wordVectors.npy')


def get_train_batch():
    """
    由于有12500个pos，12500个neg。取前11500个pos和后11500个neg做训练集，
    中间2000个(1000 pos，1000 neg)做测试集
    label中[1, 0]表示pos，[0, 1]表示neg
    :return:
    """
    labels_ = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        if i % 2 == 0:
            num = randint(0, 11499)
            labels_.append([1, 0])  # 积极
        else:
            num = randint(13500, 24999)
            labels_.append([0, 1])  # 消极
        arr[i] = ids[num]
    return arr, labels_


def get_test_batch():
    labels_ = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        num = randint(11500, 13499)
        if num <= 12499:
            labels_.append([1, 0])
        else:
            labels_.append([0, 1])
        arr[i] = ids[num]
    return arr, labels_

def train():
    # 定义输入特征和label
    labels = tf.placeholder(tf.float32, [batch_size, num_labels])
    input_data = tf.placeholder(tf.int32, [batch_size, max_seq_num])
    # 将输入的特征转换成词向量表示
    data = tf.nn.embedding_lookup(word_vectors, input_data)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_units)  # 定义LSTM单元
    # 定义LSTM单元的dropout
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
    # value保存顶层LSTM每一步的输出
    value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstm_units, num_labels]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_labels]))
    value = tf.transpose(value, [1, 0, 2])  # 转置，交换第0维和第1维，即行和列
    # 提取value最后一行，即LSTM最后的隐藏状态
    last = tf.gather(value, int(value.get_shape()[0]) - 1)

    # 定义输出，计算预测值
    prediction = tf.matmul(last, weight) + bias
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 定义损失函数，使用softmax交叉熵作为损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))

    # 定义优化器
    optimizer = tf.train.AdamOptimizer().minimize(loss)


    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for k in range(iterations):
            next_batch, next_batch_label = get_train_batch()
            acc, _ = sess.run([accuracy, optimizer],{input_data: next_batch, labels: next_batch_label})
            print("iteration %d, accuracy: %f" % (k, acc))
            if (k % 10000 == 0 and k != 0) or (k == iterations - 1):
                save_path = saver.save(sess, './data/sentiment.ckpt',global_step=k)
    print("saved to %s" % save_path)

# 不会加载用于测试
def restore_model_ckpt(ckpt_file_path):
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./data/sentiment.ckpt-40000.meta')  # 加载模型结构
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_file_path))  # 只需要指定目录就可以恢复所有变量信息

    # 直接获取保存的变量
    print(sess.run('b:0'))

    # 获取placeholder变量
    input_x = sess.graph.get_tensor_by_name('x:0')
    input_y = sess.graph.get_tensor_by_name('y:0')
    # 获取需要进行计算的operator
    op = sess.graph.get_tensor_by_name('op_to_store:0')
    '''
    
    # 加入新的操作
    add_on_op = tf.multiply(op, 2)

    ret = sess.run(add_on_op, {input_x: 5, input_y: 5})
    print(ret)
    '''

if __name__=='__main__':
    restore_model_ckpt()