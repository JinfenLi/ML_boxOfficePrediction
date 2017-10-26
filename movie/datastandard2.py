import numpy as np
import pandas as pd
import xlrd
import os
import sys
import codecs
import ast
import csv
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest ,chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats

class ActorInfo(object):
    def __init__(self):
        self.workbook = xlrd.open_workbook('movie.xlsx')

    def readActorInfo(self):
        table = self.workbook.sheet_by_index(6)  # 通过索引顺序获取工作表
        self.actor = {}

        for i in range(table.nrows):
            self.actor[table.cell(i, 0).value] = table.cell(i, 3).value

        print(self.actor["埃迪·雷德梅恩"])
        return self.actor

    def readMovieType(self):
        table = self.workbook.sheet_by_index(7)
        self.type = {}

        for i in range(table.nrows):
            self.type[table.cell(i, 0).value] = table.cell(i, 4).value

        return self.type

    def readMovieInfo(self,num):

        table = self.workbook.sheet_by_index(num)

        if(num==0):
            f = f = open("traindata.txt", "w")
        elif(num==1):
            f = open("testdata.txt", "w")

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

    def readtxt(self,num):

        if(num==0):
            txt = "traindata.txt"
        elif(num==1):
            txt = "testdata.txt"

        data = np.loadtxt(txt, delimiter=u"\t", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))

        director = np.loadtxt(txt, delimiter=u"\t", usecols=(13,), dtype=str)
        actors = np.loadtxt(txt, delimiter=u"\t", usecols=(14,), dtype=str)
        types = np.loadtxt(txt, delimiter=u"\t",usecols=(15),dtype=str)
        tdata = data.T
        dir = []
        for i in range(len(director)):

            dir.append(float(director[i]))

        d = np.array(dir)
        newdir = []
        dmax = d.max()
        dmin = d.min()
        dmu = np.average(d)
        dsigma = d.std()



        for a in dir:

            # if (a != 0):
            #     b = sigmoid(a, 1)
            # else:
            #     b = float(0)
            b = maxMinNormalization(a,dmin,dmax)
            newdir.append(b)
        newdir = np.array(newdir)
        print("newdir", newdir)

        aclist = []
        tmptac = []
        newac = []

        for a in actors:
            ac = str(a).split(",")
            #由于最后一个空字符也被转化，所以去除最后一个
            ac = np.array(ac)[:-1]

            for act in ac:

                aclist.append(float(act))

            tmptac.append(np.average(aclist))

        tmptac = np.array(tmptac)
        amu = np.average(tmptac)
        asigma = tmptac.std()
        amin = np.min(tmptac)
        amax = np.max(tmptac)

        for a in tmptac:
            # b = sigmoid(a,1)
            b = maxMinNormalization(a,amin,amax)
            newac.append(b)
        newac = np.array(newac)
        print("newac",newac)

        typelist = []
        tmpttype = []
        newtype = []

        for a in types:
            tp = str(a).split(",")
            tp = np.array(tp)[:-1]

            for t in tp:
                typelist.append(float(t))

            tmpttype.append(np.average(typelist))

        tmpttype = np.array(tmpttype)
        tmin = np.min((tmpttype))
        tmax = np.max((tmpttype))
        for a in tmpttype:
            # b = sigmoid(a,1)
            b = maxMinNormalization(a,tmin,tmax)
            newtype.append(b)
        newtype = np.array(newtype)
        print("newtype",newtype)

        boxoffice = tdata[0]
        sortboxoffice = sorted(boxoffice)
        length = len(sortboxoffice)
        firtbxf = sortboxoffice[length//10-1]
        secondbxf = sortboxoffice[2*length//10-1]
        thirdbxf = sortboxoffice[3*length//10-1]
        fourthbxf = sortboxoffice[4*length//10-1]
        fifthbxf = sortboxoffice[5*length//10-1]
        sixthbxf = sortboxoffice[6*length//10-1]
        seventhbxf = sortboxoffice[7*length//10-1]
        eighthbxf = sortboxoffice[8*length//10-1]
        ninthbxf = sortboxoffice[9*length//10-1]
        tenthbxf = sortboxoffice[length-1]


        for i in range(len(boxoffice)):
            if(boxoffice[i]<=firtbxf):
                boxoffice[i] = 0
            elif(boxoffice[i]>firtbxf and boxoffice[i]<=secondbxf):
                boxoffice[i]=1
            elif(boxoffice[i]>secondbxf and boxoffice[i]<=thirdbxf):
                boxoffice[i] = 2
            elif (boxoffice[i] > thirdbxf and boxoffice[i] <= fourthbxf):
                boxoffice[i] = 3
            elif (boxoffice[i] > fourthbxf and boxoffice[i] <= fifthbxf):
                boxoffice[i] = 4
            elif (boxoffice[i] > fifthbxf and boxoffice[i] <= sixthbxf):
                boxoffice[i] = 5
            elif (boxoffice[i] > sixthbxf and boxoffice[i] <= seventhbxf):
                boxoffice[i] = 6
            elif (boxoffice[i] > seventhbxf and boxoffice[i] <= eighthbxf):
                boxoffice[i] = 7
            elif (boxoffice[i] > eighthbxf and boxoffice[i] <= ninthbxf):
                boxoffice[i] = 8
            elif (boxoffice[i] > ninthbxf and boxoffice[i] <= tenthbxf):
                boxoffice[i] = 9

        print("newbox",boxoffice)


        y = np.array(tdata[1])
        yearmax = 2017
        yearmin = 2014
        for i in range(len(y)):
            y[i] = maxMinNormalization(y[i],yearmin,yearmax)
        print("newyear",y)

        ry = np.array(tdata[4])
        rymax = ry.max()
        rymin = ry.min()
        for i in range(len(ry)):
            ry[i] = maxMinNormalization(ry[i], rymin, rymax)
        print("newreday",ry)

        compete = np.array(tdata[5])
        competemax = compete.max()
        competemin = compete.min()
        compete = maxminmatrix(compete,competemin,competemax)
        print("newcompete",compete)

        cost = np.array(tdata[8])
        costmax = cost.max()
        costmin = cost.min()
        cost = maxminmatrix(cost,costmin,costmax)
        print("newcost", cost)

        monthsearch = np.array(tdata[9])
        monthsearchmax = monthsearch.max()
        monthsearchmin = monthsearch.min()
        monthsearch = maxminmatrix(monthsearch,monthsearchmin,monthsearchmax)
        print("newmonthsearch",monthsearch)

        weeksearch = np.array(tdata[10])
        weeksearchmax = weeksearch.max()
        weeksearchmin = weeksearch.min()
        weeksearch = maxminmatrix(weeksearch,weeksearchmin,weeksearchmax)
        print("weeksearch", weeksearch)

        volume = np.array(tdata[12])
        volumemax = volume.max()
        volumemin = volume.min()
        volume = maxminmatrix(volume,volumemin,volumemax)
        print("volume",volume)

        totaldata = []
        matrix = np.vstack((boxoffice,y,tdata[2],tdata[3],ry,compete,tdata[6],tdata[7],cost,monthsearch,weeksearch,tdata[11]
                   ,volume,newdir,newac,newtype))
        totaldata = matrix.T
        print(len(totaldata))

        #写入cvs
        if (num == 0):
            file = "traindata.csv"
        elif (num == 1):
            file = "testdata.csv"
        csvFile2 = open(file, 'w', newline='')  # 设置newline，否则两行之间会空一行
        writer = csv.writer(csvFile2)
        m = len(totaldata)
        # writer.writerow("boxoffice,year,month,day,reday,compete,ip,series,cost,monthsearch,weeksearch,import,volume,director,actor,type")
        for i in range(m):
            writer.writerow(totaldata[i])
        csvFile2.close()

        self.totaldata = totaldata

    def feature_selection(self):

        data = pd.DataFrame(pd.read_csv('traindata.csv'))

        #选取特征和目标
        x = np.array(data[['year','month','day','reday','compete','ip','series','cost','monthsearch'
                           ,'weeksearch','import','volume','director','actor','type']])
        y = np.array(data['boxoffice'])
        print(x)
        #范围0-1缩放标准化
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaler = min_max_scaler.fit_transform(x)
       # print(X_scaler)
       #  #移除方差较低的特征
       #  sel = VarianceThreshold(threshold=0.08)
       #  x_sel = sel.fit_transform(x_scaler)
        #print(x_sel.shape)



        #选择相关性最高的前12个特征
        x_chi2 = SelectKBest(chi2, k =12).fit_transform(x,y)
        print("chi2",x_chi2)

        # pca = PCA(n_components=3)
        # X_std_pca = pca.fit_transform(x_scaler)
        # print(X_std_pca.shape)
        # print(pca.explained_variance_)

        pca = PCA(n_components='mle')
        X_std_pca = pca.fit_transform(x_scaler)
        print("pca.explained_variance_ratio",pca.explained_variance_ratio_)
        print("pca.explained_variance",pca.explained_variance_)

        return x_chi2

















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

def maxminmatrix(x,min,max):
    for i in range(len(x)):
        x[i] = maxMinNormalization(x[i],min,max)
    return x