import numpy as np
import math
import random

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


class NN(object):
    def __init__(self, ni, nh, no):
        # 结点数
        self.ni = ni + 1
        self.nh = nh
        self.no = no

        # 值
        self.ai = np.ones((self.ni,))
        self.ah = np.ones((self.nh,))
        self.ao = np.ones((self.no,))
        # self.ai = [1.] * self.ni
        # self.ah = [1.] * self.nh
        # self.ao = [1.] * self.no

        # 权重
        self.wi = np.random.uniform(-0.2, 0.2, self.ni * self.nh).reshape(self.ni, self.nh)
        self.wo = np.random.uniform(2., -2., self.nh * self.no).reshape(self.nh, self.no)
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
        self.ci = np.zeros((self.ni, self.nh))
        self.co = np.zeros((self.nh, self.no))
        # self.ci = makeMatrix(self.ni, self.nh)
        # self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        assert (len(inputs) == self.ni - 1)
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        for i in range(self.nh):
            s = 0.
            for j in range(self.ni):
                s += self.ai[j] * self.wi[j][i]
            self.ah[i] = sigmoid(s)

        for i in range(self.no):
            s = 0.
            for j in range(self.nh):
                s += self.ah[j] * self.wo[j][i]
            self.ao[i] = sigmoid(s)

    def back_propagate(self, targets, N, M):
        assert (len(targets) == self.no)

        # y = sigmoid(a2 + b), J = 0.5 * (y - t) ** 2, delta_J = (y - t) * y ' * h
        # output_delta = (y - t) * y'
        output_deltas = np.zeros(self.no)
        # output_deltas = [0.] * self.no
        # print(output_deltas)
        for i in range(self.no):
            err = targets[i] - self.ao[i]
            output_deltas[i] = dsigmoid(self.ao[i]) * err

            # hidden_delta = (y - t) * y' * Wo * h'
        # delta_J = (y - t) * y' * Wo * h' * ai
        hidden_deltas = np.zeros(self.nh)
        # hidden_deltas = [0.] * self.nh
        # print(hidden_deltas)
        for i in range(self.nh):
            err = 0.
            for j in range(self.no):
                err += output_deltas[j] * self.wo[i][j]
            hidden_deltas[i] = dsigmoid(self.ah[i]) * err

            # 这里取两次的delta来逐步改变
        # delta_j = (y - t) * y' * ah
        # W_new = W_old + r1 * delta_J + r2 * delta_J_old
        for i in range(self.nh):
            for j in range(self.no):
                change = output_deltas[j] * self.ah[i]
                self.wo[i][j] += N * change + M * self.co[i][j]
                self.co[i][j] = change

                # 更新input weight
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] += N * change + M * self.ci[i][j]
                self.ci[i][j] = change

                # 计算错误率
        err = 0.
        for i in range(len(targets)):
            err += 0.5 * (targets[i] - self.ao[i]) ** 2
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

    def weights(self):
        print("\nInput weights:")
        for i in range(self.ni):
            print(self.wi[i])
        print("\nOutput weights:")
        for i in range(self.nh):
            print(self.wo[i])


def main():
    pat = np.array([
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ])
    nn = NN(2, 2, 1)
    # nn.weights()
    nn.train(pat)


if __name__ == "__main__":
    main()