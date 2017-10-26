import numpy as np
import pandas as pd
import xlrd
import csv
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA


class DataInfo(object):
    def __init__(self):
        self.workbook = xlrd.open_workbook('movie.xlsx')
    # 取得演员标签
    def readActorInfo(self):
        table = self.workbook.sheet_by_index(6)  # 通过索引顺序获取工作表
        self.actor = {}

        for i in range(table.nrows):
            self.actor[table.cell(i, 0).value] = table.cell(i, 3).value

        return self.actor
    # 取得电影类型标签
    def readMovieType(self):
        table = self.workbook.sheet_by_index(7)
        self.type = {}

        for i in range(table.nrows):
            self.type[table.cell(i, 0).value] = table.cell(i, 4).value

        return self.type

    def readMovieInfo(self,num):

        table = self.workbook.sheet_by_index(num)

        if(num==0):
            f = open("traindatabysix.txt", "w")
        elif(num==1):
            f = open("testdatabysix.txt", "w")

        for rowNum in range(table.nrows):
            tmp = ""
            for colNum in range(1, table.ncols):
                if table.cell(rowNum, colNum).value != None:

                    if (colNum == 14):
                        actor = str(table.cell(rowNum, colNum).value)
                        if actor in self.actor:
                            actor = self.actor[actor]
                            tmp += str(actor) + u"\t"
                        else:
                            tmp += "2.0" + u"\t"

                    elif (colNum == 15):
                        actor = str(table.cell(rowNum, colNum).value)
                        print(actor)
                        actors = actor.split(" ")
                        for a in actors:
                            if a in self.actor:
                                a = self.actor[a]
                                tmp += str(a) + ","
                            else:
                                tmp += "2.0" + ","


                    elif (colNum == 16):
                        tmp += u"\t"
                        type = str(table.cell(rowNum, colNum).value)
                        types = type.split(" ")
                        for t in types:
                            if t in self.type:
                                t = self.type[t]
                                tmp += str(t) + ","
                            else:
                                continue

                    else:
                        tmp += str(table.cell(rowNum, colNum).value) + u"\t"

            tmp += "\n"
            f.write(tmp)
        f.close()

    def datastandard(self):

        txt = "data.txt"
        data = np.loadtxt(txt, delimiter=u"\t", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
        director = np.loadtxt(txt, delimiter=u"\t", usecols=(13,), dtype=str)
        actors = np.loadtxt(txt, delimiter=u"\t", usecols=(14,), dtype=str)
        types = np.loadtxt(txt, delimiter=u"\t",usecols=(15),dtype=str)
        tdata = data.T
        dir = []

        for i in range(len(director)):
            dir.append(float(director[i]))

        aclist = []
        newac = []

        for a in actors:
            ac = str(a).split(",")
            #由于最后一个空字符也被转化，所以去除最后一个
            ac = np.array(ac)[:-1]

            for act in ac:
                aclist.append(float(act))

            newac.append(np.average(aclist))

        typelist = []
        newtype = []

        for a in types:
            tp = str(a).split(",")
            tp = np.array(tp)[:-1]

            for t in tp:
                typelist.append(float(t))

            newtype.append(np.average(typelist))

        boxoffice = tdata[0]
        tenboxoffice = boxofficebyten(boxoffice)
        sixboxoffice = boxofficebysix(boxoffice)
        matrix = np.vstack((tdata[1:],dir,newac,newtype))

        totaldata = np.vstack((tenboxoffice,maxminmatrix(matrix.T).T)).T
        #写入cvs
        file = "data2.csv"
        csvFile2 = open(file, 'w', newline='')  # 设置newline，否则两行之间会空一行
        writer = csv.writer(csvFile2)
        # writer.writerow("boxoffice,year,month,day,reday,compete,ip,series,cost,monthsearch,weeksearch,import,volume,director,actor,type")
        for i in range(len(totaldata)):
            writer.writerow(totaldata[i])
        csvFile2.close()
        self.totaldata = totaldata

    def feature_selection(self):

        data = pd.DataFrame(pd.read_csv('data2.csv'))

        #选取特征和目标
        x = np.array(data[['year','month','day','reday','compete','ip','series','cost','monthsearch'
                           ,'weeksearch','import','volume','director','actor','type']])
        y = np.array(data['boxoffice'])
        #范围0-1缩放标准化
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaler = min_max_scaler.fit_transform(x)
       # print(X_scaler)
       #  #移除方差较低的特征
       #  sel = VarianceThreshold(threshold=0.08)
       #  x_sel = sel.fit_transform(x_scaler)

        #选择相关性最高的前12个特征,从数据看去除了竞争量，月搜索，演员
        feature = SelectKBest(chi2, k =13).fit_transform(x,y)

        # pca = PCA(n_components=3)
        # X_std_pca = pca.fit_transform(x_scaler)
        # print(X_std_pca.shape)
        # print(pca.explained_variance_)

        # pca = PCA(n_components='mle')
        # X_std_pca = pca.fit_transform(x_scaler)
        # print("pca.explained_variance_ratio",pca.explained_variance_ratio_)
        # print("pca.explained_variance",pca.explained_variance_)

        data = np.vstack((y,feature.T)).T

        return data


def boxofficebyten(boxoffice):
    sortboxoffice = sorted(boxoffice)
    length = len(sortboxoffice)
    firtbxf = sortboxoffice[2*length // 10 - 1]
    secondbxf = sortboxoffice[4 * length // 10 - 1]
    thirdbxf = sortboxoffice[6 * length // 10 - 1]
    fourthbxf = sortboxoffice[8 * length // 10 - 1]
    fifthbxf = sortboxoffice[length - 1]
    # sixthbxf = sortboxoffice[6 * length // 10 - 1]
    # seventhbxf = sortboxoffice[7 * length // 10 - 1]
    # eighthbxf = sortboxoffice[8 * length // 10 - 1]
    # ninthbxf = sortboxoffice[9 * length // 10 - 1]
    # tenthbxf = sortboxoffice[length - 1]

    for i in range(len(boxoffice)):
        if (boxoffice[i] <= firtbxf):
            boxoffice[i] = 0
        elif (boxoffice[i] > firtbxf and boxoffice[i] <= secondbxf):
            boxoffice[i] = 1
        elif (boxoffice[i] > secondbxf and boxoffice[i] <= thirdbxf):
            boxoffice[i] = 2
        elif (boxoffice[i] > thirdbxf and boxoffice[i] <= fourthbxf):
            boxoffice[i] = 3
        elif (boxoffice[i] > fourthbxf and boxoffice[i] <= fifthbxf):
            boxoffice[i] = 4
        elif (boxoffice[i] > fifthbxf):
            boxoffice[i] = 5
        # elif (boxoffice[i] > sixthbxf and boxoffice[i] <= seventhbxf):
        #     boxoffice[i] = 6
        # elif (boxoffice[i] > seventhbxf and boxoffice[i] <= eighthbxf):
        #     boxoffice[i] = 7
        # elif (boxoffice[i] > eighthbxf and boxoffice[i] <= ninthbxf):
        #     boxoffice[i] = 8
        # elif (boxoffice[i] > ninthbxf and boxoffice[i] <= tenthbxf):
        #     boxoffice[i] = 9

        return boxoffice

def boxofficebysix(x):

    for i in range(len(x)):
        if x[i]<=10000:
            x[i] = 0
        elif x[i]>10000 and x[i]<=20000:
            x[i] = 1
        elif x[i]>20000 and x[i]<=30000:
            x[i] = 2
        elif x[i]>30000 and x[i]<=40000:
            x[i] = 3
        elif x[i]>40000 and x[i]<=50000:
            x[i] = 4
        elif x[i] > 50000 and x[i] <= 100000:
            x[i] = 5
        elif x[i] > 100000 and x[i] <= 200000:
            x[i] = 6
        elif x[i] > 200000:
            x[i] = 7
    return x











def maxMinNormalization(x, min, max):
    x = (x - min) / (max - min)
    return x


def Z_ScoreNormalization(x, mu, sigma):
    x = (x - mu) / sigma;
    return x;


def sigmoid(X, useStatus):
    if useStatus:
        return 1.0 / (1 + np.exp(-float(X)));
    else:
        return float(X);

def sigmoidmatrix(x):
    for i in range(len(x)):
        x[i] = sigmoid(x[i],1)
    return x

def maxminmatrix(x):
    # x是原始数据
    tx = x.T
    for i in range(len(tx)):
        max = np.max(tx[i])
        min = np.min(tx[i])
        for j in range(len(tx[i])):
            tx[i][j] = (tx[i][j] - min)/(max - min)

    return tx.T

