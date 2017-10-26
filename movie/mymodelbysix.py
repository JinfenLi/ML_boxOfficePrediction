import numpy as np
import tensorflow as tf
import pandas as pd
import os
import datastandard




def gen_train_data(train_size = 190, out_dim = 8):

    data = datastandard.DataInfo().feature_selection()
    # noise = np.random.normal(0, 0.00000000000005, data.shape)  # 噪点
    # data = data + noise
    np.random.shuffle(data)
    # 存储标签
    pre_mat = []
    for i in range(len(data)):
        pre = np.zeros([out_dim], dtype=np.float32)
        pre[int(data[i][0])] = 1
        pre_mat.append(pre)

    pre_mat = np.asarray(pre_mat)

    train_data = data[0:train_size, 1:]
    train_label = pre_mat[0:train_size, :]
    vali_data = data[train_size:, 1:]
    vali_label = pre_mat[train_size:, :]
    return train_data, train_label,vali_data, vali_label
import datawrapper


# def gen_train_data(train_size = 190, out_dim = 10, use17=False):
#     training_data, validation_data, test_data = datawrapper.load_data_wrapper()
#     data = np.vstack((training_data,validation_data))
#     if not use17:
#         np.random.shuffle(data)
#     # 存储标签
#     pre_mat = []
#     for i in range(len(data)):
#         pre = np.zeros([out_dim], dtype=np.float32)
#         pre[int(data[i][0])] = 1
#         pre_mat.append(pre)
#
#     pre_mat = np.asarray(pre_mat)
#     if use17:
#         train_data = []
#         train_label = []
#         vali_data = []
#         vali_label = []
#         for i in range(len(data)):
#             # 17年的当验证集
#             if data[i][1] == 1.0:
#                 vali_data.append(data[i][1:])
#
#                 vali_label.append(pre_mat[i])
#             # 14-16年的当测试集
#             else:
#                 train_data.append(data[i][1:])
#                 train_label.append(pre_mat[i])
#         return np.asarray(train_data), np.asarray(train_label),np.asarray(vali_data), np.asarray(vali_label)
#     else:
#         train_data = data[0:train_size, 1:]  # slice the data into data of train and test
#         train_label = pre_mat[0:train_size, :]
#         vali_data = data[train_size:, 1:]
#         vali_label = pre_mat[train_size:, :]
#         return train_data, train_label,vali_data, vali_label

class MLP(object):
    def __init__(self, sess, name="default", input_dim=20, output_dim=8, hidden_depth=8, stddev=0.02,
                 learning_rate=0.008, scope=None):
        self.sess = sess
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_depth = hidden_depth
        self.stddev = stddev
        self.learning_rate = learning_rate

        self.train_data, self.train_label, self.vali_data, self.vali_label = None, None, None, None
        self.train_step, self.accuracy, self.x, self.y, self.loss, self.y_hat = None, None, None, None, None, None
        self.scope = scope
        self.data = None
    # 此函数主要作用是取得输入维数
    def get_data(self, train_size=190):

        self.train_data, self.train_label, self.vali_data, self.vali_label = gen_train_data(train_size,self.output_dim)
        self.input_dim = len(self.train_data[0])


    def build_MLP(self):
        with tf.name_scope(self.name + "input_layer"):
            x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name=self.name + "x_inputs")
            # tf.nn.dropout(x, keep_prob=0.5, noise_shape=None, seed=None, name=None)
            #由于下面x会被替换，所以用x_代表最初的输入
            x_ = x
            y = tf.placeholder(tf.float32, shape=[None, self.output_dim])
            tf.summary.histogram(self.name + "target", tf.argmax(y,1))
        # in_ = self.input_dim
        with tf.name_scope(self.name + "hidden_layer"):
            # 前几层用tahn函数，最后一层用softmax
            for i in range(self.hidden_depth-1):
                with tf.name_scope(self.name + "weights"):
                    w = tf.get_variable(
                        name = self.name + "weight" + str(i),
                        dtype=tf.float32,
                        shape=[self.input_dim,self.input_dim],
                        initializer=tf.random_normal_initializer(stddev=self.stddev)
                    )
                with tf.name_scope(self.name + "bias"):
                    b = tf.get_variable(
                        name=self.name + "bias" + str(i),
                        dtype=tf.float32,
                        shape=[self.input_dim],
                        initializer=tf.zeros_initializer()
                    )
                with tf.name_scope(self.name + "output"):
                    # x = tf.nn.relu(tf.matmul(x, w) + b, name=self.name + "out_put" + str(i))
                    x = tf.nn.tanh(tf.matmul(x, w) + b, name=self.name + "out_put" + str(i))
                    # x = tf.nn.dropout(x, keep_prob=0.5)
                    # in_ = round(math.sqrt(in_ * self.output_dim))
            with tf.name_scope(self.name + "weights"):
                w = tf.get_variable(
                    name=self.name + "weight" + str(self.hidden_depth - 1),
                    dtype=tf.float32,
                    shape=[self.input_dim, self.output_dim],
                    initializer=tf.random_normal_initializer(stddev=self.stddev)
                )
            with tf.name_scope(self.name + "bias"):
                b = tf.get_variable(
                    name=self.name + "bias" + str(self.hidden_depth - 1),
                    dtype=tf.float32,
                    shape=[self.output_dim],
                    initializer=tf.zeros_initializer()
                )

            with tf.name_scope(self.name + "output"):
                y_hat = tf.nn.softmax(tf.matmul(x, w) + b, name=self.name + "softmax_output")
                tf.summary.histogram(self.name + "y_hat", y_hat)

        with tf.name_scope(name=self.name + "output_layer"):
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1]),
                                        name=self.name + "cross_entropy")
            # loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
            tf.summary.scalar(self.name + "loss", loss)
        with tf.name_scope(name=self.name + "train_step"):
            # 梯度下降
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
            # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        with tf.name_scope(name=self.name + "predict"):
            predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1), name=self.name + "predict")
            accuracy = tf.reduce_mean(tf.cast(predict, tf.float32), name=self.name + "accuracy")
            tf.summary.scalar(self.name + "accuracy", accuracy)
        return train_step, accuracy, x_, y, loss, y_hat

    def init(self):

        self.get_data(train_size=235)
        self.train_step, self.accuracy, self.x, self.y, self.loss, self.y_hat = self.build_MLP()

    def run(self, epochs=20000, name="", train_size=190):

        self.train_data, self.train_label, self.vali_data, self.vali_label = gen_train_data(
            train_size=train_size,out_dim=self.output_dim)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        #在scope范围内监控graph
        merged = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, self.scope))
        # 区别：sess.graph 把所有框架加载到一个文件中放到文件夹"logs/"里
        # 接着打开terminal，进入你存放的文件夹地址上一层，运行命令 tensorboard --logdir=logs
        # 会返回一个地址，然后用浏览器打开这个地址，在 graph 标签栏下打开
        writer = tf.summary.FileWriter("logs/", self.sess.graph)
        saver = tf.train.Saver()
        #最大准确率
        max_ = 0.0
        # 每100次训练的准确率
        acc_ = 0.0
        # 储存前一次的训练精度，用于是否收敛检验
        pre_vali_acc = 0.0
        y_hat = None
        for i in range(epochs):
            # self.sess.run(self.train_step, feed_dict={self.x:self.train_data, self.y : self.train_label})
            if i % 200 == 0:
                # 验证集的准确率
                vali_acc = self.sess.run(self.accuracy, feed_dict={self.x: self.vali_data, self.y: self.vali_label})
                y_hat = self.sess.run(self.y_hat, feed_dict={self.x: self.vali_data, self.y: self.vali_label})
                writer.add_summary(self.sess.run(merged, feed_dict={self.x: self.vali_data, self.y: self.vali_label}),i)
                acc_ = vali_acc
                if vali_acc > max_:
                    max_ = vali_acc
                    if not os.path.exists("models/" + self.name + "/" + name):
                        os.mkdir("models/" + self.name + "/" + name)
                    saver.save(self.sess, "models/" + self.name + "/" + name + "/" + "steps_" + str(i))
                #训练集的准确率
                train_acc = self.sess.run(self.accuracy, feed_dict={self.x: self.train_data, self.y: self.train_label})

                if pre_vali_acc == vali_acc :
                    break
                else:
                    pre_vali_acc = vali_acc
                print("the epochs, loss, train_accuracy, vali_accuracy and maximum accuracy is :%d, %f, %f, %f, %f" % (
                    i,
                    self.sess.run(self.loss, feed_dict={self.x: self.train_data, self.y: self.train_label}),
                    train_acc,
                    vali_acc,
                    max_
                ))
            self.sess.run(self.train_step, feed_dict={self.x: self.train_data, self.y: self.train_label})

        #返回迭代到最后一次的验证集的准确度，和预测的验证集的标签
        return acc_, y_hat





config = tf.ConfigProto()
config.gpu_options.allow_growth = True
def main():

    train_size_descent()


def train_size_descent():
    with tf.name_scope("train_size_test") as scope:
        model = MLP(tf.Session(config=config), name="train_size_test", output_dim=8, scope=scope)
        model.init()
        if not os.path.exists("models/train_size_test"):
            os.mkdir("models/train_size_test")
        acc = []
        for i in range(2):
            a = []
            for j in range(20):
                train_size = 200 + i * 10
                acc_, y_ = model.run(20000, train_size=train_size, name="train_size_" + str(train_size))
                a.append(acc_)
            acc.append(np.average(a))
        print(acc)

if __name__ == "__main__":
    main()