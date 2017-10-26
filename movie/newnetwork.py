import numpy as np
import math
import random
import  datawrapper
import tensorflow as tf

np.random.seed(0)


# random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b - a) * random.random() + a


# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


def sigmoid(x):
    # f(z) = (e ^ z - e ^ (-z)) / (e ^ z + e ^ (-z))
    # http://ufldl.stanford.edu/wiki/index.php/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C
    return math.tanh(x)


def dsigmoid(y):
    # 导数为：f'(z) = 1 - f(z) ^ 2
    return 1.0 - y ** 2

def softmax(x):
 return np.exp(x)/np.sum(np.exp(x),axis=0)

def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

def loss(x,y):
    loss=0.
    for i in range(len(x)):
        if(x[i]==0):
            continue
        else:
            loss += np.log(x[i]) * y[i]

    return loss
#theta.shape==(k,n+1)
#lenda是正则化系数/权重衰减项系数，alpha是学习率
def J(X,classLabels,theta,alpha,lenda):
    bin_classLabels=label_binarize(classLabels,classes=np.unique(classLabels).tolist()).reshape((m,k))  #二值化 (m*k)
    dataSet=np.concatenate((X,np.ones((m,1))),axis=1).reshape((m,n+1)).T   #转换为（n+1,m）

    theta_data=theta.dot(dataSet)  #(k,m)
    theta_data = theta_data - np.max(theta_data)   #k*m
    prob_data = np.exp(theta_data) / np.sum(np.exp(theta_data), axis=0)  #(k*m)
    #print(bin_classLabels.shape,prob_data.shape
    cost = (-1 / m) * np.sum(np.multiply(bin_classLabels,np.log(prob_data).T)) + (lenda / 2) * np.sum(np.square(theta))  #标量
    #print(dataSet.shape,prob_data.shape)
    grad = (-1 / m) * (dataSet.dot(bin_classLabels - prob_data.T)).T + lenda * theta  #(k*N+1)

    return cost,grad
class NN(object):
    def __init__(self, ni, nh, no):
        # 结点数
        self.ni = ni
        self.nh = nh
        self.no = no

        # 值
        self.ai = np.ones((self.ni,))
        self.ah = np.ones((self.nh,))
        self.ao = np.zeros((self.no,))
        # self.ai = [1.] * self.ni
        # self.ah = [1.] * self.nh
        # self.ao = [1.] * self.no

        # 权重
        self.wi = np.random.uniform(-0.2, 0.2, self.ni * self.nh).reshape(self.nh, self.ni)
        self.wo = np.random.uniform(2., -2., self.nh * self.no).reshape(self.no, self.nh)

        #偏值
        self.bi = np.random.uniform(-0.2, 0.2, self.ni * self.nh).reshape(self.nh, self.ni)
        self.bo = np.random.uniform(2., -2., self.nh * self.no).reshape(self.no, self.nh)
        # self.wi = np.zeros((self.ni, self.nh))
        # self.wo = np.zeros((self.nh, self.no))
        # self.wi = makeMatrix(self.ni, self.nh)
        # self.wo = makeMatrix(self.nh, self.no)
        #
        # for i in range(self.ni):
        #     for j in range(self.nh):
        #         self.wi[i][j] = rand(-0.2, 0.2)
        #         # print(type(self.wi[i][j]))
        #
        # for i in range(self.nh):
        #     for j in range(self.no):
        #         self.wo[i][j] = rand(-2.0, 2.0)

        # 旧的weight
        self.ci = np.zeros((self.nh, self.ni))
        self.co = np.zeros((self.no, self.nh))
        # self.ci = makeMatrix(self.ni, self.nh)
        # self.co = makeMatrix(self.nh, self.no)
        #旧的bias
        self.di = np.zeros((self.nh, self.ni))
        self.do = np.zeros((self.no, self.nh))

    def update(self, inputs):

        #assert (len(inputs) == self.ni )
        for i in range(self.ni ):
            self.ai[i] = inputs[i]
        #ah为隐藏层的输出值
        for i in range(self.nh):
            s = 0.
            for j in range(self.ni):
                s += self.ai[j] * self.wi[i][j]
            self.ah[i] = sigmoid(s)

        for i in range(self.no):
            s = 0.
            for j in range(self.nh):
                s += self.ah[j] * self.wo[i][j]
            self.ao[i] = sigmoid(s)
        self.ao = softmax(self.ao)

    def feedforward(self, a):
        """
        前向传输计算每个神经元的值
        :param a: 输入值
        :return: 计算后每个神经元的值
        """
        for b, w in zip(self.biases, self.weights):
            # 加权求和以及加上 biase
            a = sigmoid(np.dot(w, a) + b)
        return a

    def back_propagate(self, targets, N, M):
#        assert (len(targets) == self.no)

        # y = sigmoid(a2 + b), J = 0.5 * (y - t) ** 2, delta_J = (y - t) * y ' * h
        # output_delta = (y - t) * y'
        output_deltas = np.zeros(self.no)
        # output_deltas = [0.] * self.no
        # print(output_deltas)
        #输出层残差
        for i in range(self.no):
            # err = targets[i] - self.ao[i]
            argmax = np.argmax(self.ao)

            err = int(targets[0])-1 - argmax
            if(argmax==int(targets[0]-1)):
                output_deltas[i] = (self.ao[argmax] - 1) * self.ao[argmax] * err
            else:
                output_deltas[i] = self.ao[argmax] * self.ao[argmax] * err


            # hidden_delta = (y - t) * y' * Wo * h'
        # delta_J = (y - t) * y' * Wo * h' * ai
        #隐藏层残差
        hidden_deltas = np.zeros(self.nh)
        # hidden_deltas = [0.] * self.nh
        # print(hidden_deltas)
        for i in range(self.nh):
            err = 0.
            for j in range(self.no):
                err += output_deltas[j] * self.wo[j][i]
            hidden_deltas[i] = dsigmoid(self.ah[i]) * err

            # 这里取两次的delta来逐步改变
        # delta_j = (y - t) * y' * ah
        # W_new = W_old + r1 * delta_J + r2 * delta_J_old
        for i in range(self.nh):
            for j in range(self.no):
                change = output_deltas[j] * self.ah[i]
                self.wo[j][i] += N * change + M * self.co[j][i]
                self.co[j][i] = change

                # 更新input weight
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[j][i] += N * change + M * self.ci[j][i]
                self.ci[j][i] = change

                # 计算错误率

        # err = 0.
        # for i in range(len(targets)):
        #     err += 0.5 * (targets[i] - self.ao[i]) ** 2
        # return err
        err = loss(targets,self.ao)

        return err

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        for i in range(iterations):
            err = 0.
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                err += self.back_propagate(targets, N, M)
            if i % 100 == 0:
                # self.weights()
                print('error %-.5f' % err)

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        随机梯度下降
        :param training_data: 输入的训练集
        :param epochs: 迭代次数
        :param mini_batch_size: 小样本数量
        :param eta: 学习率
        :param test_data: 测试数据集
        """
        # if test_data:
        n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            # 搅乱训练集，让其排序顺序发生变化
            random.shuffle(training_data)
            # 按照小样本数量划分训练集
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                # 根据每个小样本来更新 w 和 b，代码在下一段
                self.update_mini_batch(mini_batch, eta)
                #print("minibatch",mini_batch)
            # 输出测试每轮结束后，神经网络的准确度
            if len(test_data)>0:
                print ("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        更新 w 和 b 的值
        :param mini_batch: 一部分的样本
        :param eta: 学习率
        """
        # 根据 biases 和 weights 的行列数创建对应的全部元素值为 0 的空矩阵
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x in mini_batch:
            # 根据样本中的每一个输入 x 的其输出 y，计算 w 和 b 的偏导数
            delta_nabla_b, delta_nabla_w = self.backprop(x.T[:13].T, x.T[13].T)
            # 累加储存偏导值 delta_nabla_b 和 delta_nabla_w
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # 更新根据累加的偏导值更新 w 和 b，这里因为用了小样本，
        # 所以 eta 要除于小样本的长度
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    # def predict_(self, x):
    #     result = np.dot(self.w,x)
    #     row, column = result.shape
    #
    #     # 找最大值所在的列
    #     _positon = np.argmax(result)
    #     m, n = divmod(_positon, column)
    #
    #     return m

    def predict(self,inputs):
        labelmatrix = []
        for i in inputs:
            self.update(i)
            label = np.argmax(self.ao)+1
            labelmatrix.append(label)

        print(labelmatrix)
        return labelmatrix

    def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):



    def weights(self):
        print("\nInput weights:")
        for i in range(self.ni):
            print(self.wi[i])
        print("\nOutput weights:")
        for i in range(self.nh):
            print(self.wo[i])
def vector(i):
    vector = [0,0,0,0,0,0,0,0,0,0]
    vector[i] = 1
    return vector

def main():
    training_data, validation_data, test_data=datawrapper.load_data_wrapper()


    pat = []



    for t in training_data:
        td=[]
        td.append(t[0])
        td.append(t[1])
        td.append(t[2])
        td.append(t[3])
        td.append(t[4])
        td.append(t[5])
        td.append(t[6])
        td.append(t[7])
        td.append(t[8])
        td.append(t[9])
        td.append(t[10])
        td.append(t[11])
        td.append(t[12])




        tt=[]

        # tt=vector(int(t[13])-1)
        tt.append(t[13])

        ttt=[]
        ttt.append(td)
        ttt.append(tt)
        pat.append(ttt)







    nn = NN(13, 13, 10)
    print(test_data)
    # nn.weights()
    nn.train(pat)


    test_predict = nn.predict(test_data.T[:13].T)

if __name__ == "__main__":
    main()