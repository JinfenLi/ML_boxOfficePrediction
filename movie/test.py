import datastandard2
import datawrapper
import network
#import newnetwork
import datastandard
import numpy as np
data = datastandard2.ActorInfo()
x=[0.1,0.2]
y=[0.1]
x = np.array(x)
y = np.array(y)
#test = newnetwork.softmax_loss(x,y)
#print(test)
# data2 = datawrapper.load_data()
# data.readMovieType()
# data.readActorInfo()
# data.readMovieInfo(1)
#data.readtxt(1)
#data.feature_selection()
#print(data2)
# datawrapper.load_data_wrapper()
t = datastandard.DataInfo()
t.datastandard()
t.feature_selection()
# traindata = np.loadtxt("traindata.txt", delimiter=u"\t", usecols=(0,),dtype = str)
# testdata = np.loadtxt("testdata.txt", delimiter=u"\t", usecols=(0,))
# print(np.max(traindata))
# print(np.max(testdata))
#
# training_data, validation_data, test_data = datawrapper.load_data_wrapper()
# # 13个输入神经元，一层隐藏层，包含 30 个神经元，输出层包含 1 个神经元
# net = network.Network([13, 2,1])
# net.SGD(training_data, 30, 10, 0.01, test_data = test_data)
# Epoch 0: 9038 / 10000
# Epoch 1: 9178 / 10000
# Epoch 2: 9231 / 10000
# ...
# Epoch 27: 9483 / 10000
# Epoch 28: 9485 / 10000
# Epoch 29: 9477 / 10000