import numpy as np
import math
import random
import  datawrapper
import tensorflow as tf
import pandas as pd
import os
from sklearn import cross_validation
# use17:是否有用17年的当验证集



def gen_train_data(train_size = 190, out_dim = 10, use17=False):
    training_data, validation_data, test_data = datawrapper.load_data_wrapper()
    data = np.vstack((training_data,validation_data))
    if not use17:
        np.random.shuffle(data)
    # 存储标签
    pre_mat = []
    for i in range(len(data)):
        pre = np.zeros([out_dim], dtype=np.float32)
        pre[int(data[i][0])] = 1
        pre_mat.append(pre)

    pre_mat = np.asarray(pre_mat)
    if use17:
        train_data = []
        train_label = []
        vali_data = []
        vali_label = []
        for i in range(len(data)):
            # 17年的当验证集
            if data[i][1] == 1.0:
                vali_data.append(data[i][1:])

                vali_label.append(pre_mat[i])
            # 14-16年的当测试集
            else:
                train_data.append(data[i][1:])
                train_label.append(pre_mat[i])
        return np.asarray(train_data), np.asarray(train_label),np.asarray(vali_data), np.asarray(vali_label)
    else:
        train_data = data[0:train_size, 1:]  # slice the data into data of train and test
        train_label = pre_mat[0:train_size, :]
        vali_data = data[train_size:, 1:]
        vali_label = pre_mat[train_size:, :]
        return train_data, train_label,vali_data, vali_label

class MLP(object):
    def __init__(self, sess, name="default", input_dim=20, output_dim=10, hidden_depth=4, stddev=0.02,
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

    def get_data(self, train_size=190, use17=False):
        if not use17:
            self.train_data,self.train_label,self.vali_data,self.vali_label = gen_train_data(train_size,self.output_dim,use17=False)
        else:
            self.train_data, self.train_label, self.vali_data, self.vali_label = gen_train_data(train_size,self.output_dim,use17=True)
        self.input_dim = len(self.train_data[0])


    def build_MLP(self):
        with tf.name_scope(self.name + "input_layer"):
            x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name=self.name + "x_inputs")
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
                    x = tf.nn.tanh(tf.matmul(x, w) + b, name=self.name + "out_put" + str(i))
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
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
            # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        with tf.name_scope(name=self.name + "predict"):
            predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1), name=self.name + "predict")
            accuracy = tf.reduce_mean(tf.cast(predict, tf.float32), name=self.name + "accuracy")
            tf.summary.scalar(self.name + "accuracy", accuracy)
        return train_step, accuracy, x_, y, loss, y_hat

    def init(self, use17=False):
        if use17:
            self.get_data(use17=use17, train_size=235)
        else:
            self.get_data(use17=use17)
        self.train_step, self.accuracy, self.x, self.y, self.loss, self.y_hat = self.build_MLP()

    def run(self, epochs=20000, init_=True, name="", train_size=190, use17=False):
        if init_:
            self.init()
        self.train_data, self.train_label, self.vali_data, self.vali_label = gen_train_data(
            train_size=train_size,out_dim=self.output_dim,use17=use17)
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
        # 每50次的准确率
        acc_ = 0.0
        # 储存前一次的训练精度，用于是否收敛检验
        pre_train_acc = 0.0
        #正确的条数
        count_pre = 0
        y_ = None
        for i in range(epochs):
            self.sess.run(self.train_step, feed_dict={self.x:self.train_data, self.y : self.train_label})
            if i % 50 == 0:
                # 验证集的准确率
                vali_acc = self.sess.run(self.accuracy, feed_dict={self.x: self.vali_data, self.y: self.vali_label})
                y_ = self.sess.run(self.y_hat, feed_dict={self.x: self.vali_data, self.y: self.vali_label})
                writer.add_summary(self.sess.run(merged, feed_dict={self.x: self.vali_data, self.y: self.vali_label}),
                                   i)
                acc_ = vali_acc
                if vali_acc > max_:
                    max_ = vali_acc
                    if not os.path.exists("models/" + self.name + "/" + name):
                        os.mkdir("models/" + self.name + "/" + name)
                    saver.save(self.sess, "models/" + self.name + "/" + name + "/" + "steps_" + str(i))
                #训练集的准确率
                train_acc = self.sess.run(self.accuracy, feed_dict={self.x: self.train_data, self.y: self.train_label})

                if pre_train_acc == train_acc:
                    break
                else:
                    pre_train_acc = train_acc
                print("the epochs, loss, train_accuracy, vali_accuracy and maximum accuracy is :%d, %f, %f, %f, %f" % (
                    i,
                    self.sess.run(self.loss, feed_dict={self.x: self.train_data, self.y: self.train_label}),
                    train_acc,
                    vali_acc,
                    max_
                ))

        #返回迭代到最后一次的验证集的准确度，和预测的验证集的标签
        return acc_, y_





config = tf.ConfigProto()
config.gpu_options.allow_growth = True
def main():
    train_data, train_label,vali_data, vali_label = gen_train_data(use17=True)
    # with tf.name_scope("mnt") as scope:
    #     model = MLP(sess=tf.Session(config=config), name="model_", output_dim=10, scope=scope)
    #     model.init(use17=True)
    #     ac = []
    #
    #
    #
    #
    #     for i in range(1):
    #         acc_, y_ = model.run(name="rachel", init_=False, train_size=235, use17=True)
    #         ac.append(acc_)
    #
    #     print(ac)
    train_size_descent()


def mean_accuracy():
    acc_category = []
    for j in range(9):
        with tf.name_scope("model_" + str(j + 2)) as scope:
            model = MLP(sess=tf.Session(config=config), name="model_" + str(j + 2), output_dim=j + 2, scope=scope)
            model.init()
            if not os.path.exists("models/" + "model_" + str(j + 2)):
                os.mkdir("models/" + "model_" + str(j + 2))
            acc = []
            if j == 1:
                print(j)
            for i in range(20):
                acc_, y_ = model.run(20000, init_=False, name=str(i))
                acc.append(acc_)
            print(acc)

            acc_mean = np.average(acc)
            acc_category.append(acc_mean)
    print(acc_category)
    x = tf.placeholder(tf.float32, name="category")
    tf.summary.scalar("category", x)
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("loges", sess.graph)
    for i in range(acc_category.__len__()):
        result = sess.run(merged, feed_dict={x: acc_category[i]})
        writer.add_summary(result, i)
def train_size_descent():
    with tf.name_scope("train_size_test") as scope:
        model = MLP(tf.Session(config=config), name="train_size_test", output_dim=10, scope=scope)
        model.init()
        if not os.path.exists("models/train_size_test"):
            os.mkdir("models/train_size_test")
        acc = []

        for i in range(2):
            a = []
            for j in range(100):
                train_size = 210 + i * 10
                acc_, y_ = model.run(20000, False, train_size=train_size, name="train_size_" + str(train_size))
                a.append(acc_)
            acc.append(np.average(a))
        print(acc)
if __name__ == "__main__":
    main()