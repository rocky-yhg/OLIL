"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Perceptron Implementation ***
Paper: Bifet, Albert, et al. "Fast perceptron decision tree learning from evolving data streams."
Published in: Advances in knowledge discovery and data mining (2010): 299-310.
URL: http://www.cs.waikato.ac.nz/~eibe/pubs/Perceptron.pdf
"""

import math
import operator
import random
# import miscMethods
from collections import OrderedDict
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from classifier.classifier import SuperClassifier
from data_structures.attribute import Attribute
from dictionary.tornado_dictionary import TornadoDic
from .miscMethods import *
from sklearn import preprocessing
from itertools import zip_longest
from random import sample
class DyfAndNCD(SuperClassifier):
    """This is the initial implementation of DyfAndNCD for learning from incomplete data streams with incremental class-set."""
    ""
    LEARNER_NAME = TornadoDic.DyfAndNCD
    LEARNER_TYPE = TornadoDic.TRAINABLE
    LEARNER_CATEGORY = TornadoDic.NUM_CLASSIFIER

    __BIAS_ATTRIBUTE = Attribute()
    __BIAS_ATTRIBUTE.set_name("bias")
    __BIAS_ATTRIBUTE.set_type(TornadoDic.NUMERIC_ATTRIBUTE)
    __BIAS_ATTRIBUTE.set_possible_values(1)

    def __init__(self, labels, attributes, alpha, FR, learning_rate=1):
        super().__init__(labels, attributes)

        attributes.append(self.__BIAS_ATTRIBUTE)
        self.WEIGHTS = OrderedDict()
        self.fremove = FR
        self.remove = 0
        # self.x_attributes = OrderedDict()
        self.features_left = 0
        self.remainNum = 10
        # self.__initialize_weights()
        # self.LEARNING_RATE = learning_rate
         # self.C = 0.01
        # self.T = 0.001
        # self.gamma  = 0.00001
        self.Is_Drift = False
        self.Is_NCD_Ready = True
        # self.Is_Anomal = False
        self.alpha = alpha
        self.NCDlist = dict()
        self.clusterList = []
        self.epsilon = 0# 应该作为超参分析
        self.minPts = 10 #初始值固定
        # self.incri_of_e = 0.5#调节对新类敏感
        self.CLASSES = labels
        self.buffer = []
        self.window = 30
        self.count = 0
        self.stability= []
        self.feature_count = []
        self.alpha = 0.9 #imdb alpha = 0.6 retricted by 0.7
        self.beta = 1-self.alpha
        self.sparse = 0.1  #normal is 0.3
        self.e_step = 0.01
        np.random.seed(10)
#default value: sparse = 0.3 e_step=0.01 alpha = 0.8




    def rFeatures(self, fremove, instance):#"variable"
        #此处无标签参与
        for i in range(0,len(instance)):
            if np.random.random() > (1-fremove):
                instance[i] = 0#随机选取FR的特征用0填充
        self.update_stability(instance)
        instance = np.multiply(instance, self.feature_count)
        return instance

    def rDataTrapezoidal(self,features_left): #"Trapezoidal"
        key=0
        for a in self.ATTRIBUTES:
            if key > features_left:
                self.x_attributes[a.NAME] = 0
            else:
                self.x_attributes[a.NAME] = 1
            key+=1
        self.x_attributes["bias"] = 1

    def rDataEvolvable(self,features,flag): #"evolvable"
        key=0
        if flag == 1:
            for a in self.ATTRIBUTES:
                if key > len(self.ATTRIBUTES)-features:
                    self.x_attributes[a.NAME] = 0
                else:
                    self.x_attributes[a.NAME] = 1
                key+=1
        elif flag == 2:
            for a in self.ATTRIBUTES:
                if key > len(self.ATTRIBUTES)-features:
                    self.x_attributes[a.NAME] = 0
                else:
                    self.x_attributes[a.NAME] = 1
                key+=1
        self.x_attributes["bias"] = 1

    def ins_expand(self,x):
        arr = list()
        for a in self.ATTRIBUTES:
            arr.append(a.NAME)
        X = OrderedDict(zip_longest(arr,x))
        for key in self.x_attributes:
            if self.x_attributes[key] == 0 :
                del X[key]
        X = self.expand_space(X)
        self.update_stability(X)
        self.upKeyCount(X)
        return X

    def train(self,instance):
        # -------------
        #  Initialize NCD
        #--------------

        if self.Is_NCD_Ready:
            y = instance[-1]
            X = list(self.rFeatures(self.fremove, instance[0: len(instance)-1]))
            X.append(y)
            self.buffer.append(X)
            if len(self.buffer) > self.window:
                self._IS_READY = True
                self.Is_NCD_Ready = False
                self.DBSCAN(self.buffer)
                self.buffer = []
                print("initialize OK!!!")


    def NCD(self, x): #计算所有的距离，若有小于epsilon，则为已知类，否则为异常点,这里做修改进行简化，这里不需要知道具体是哪个类，只要是异常点就检测出来
        #检测前需要判断是否buffer满
        if len(self.buffer) > self.window:
            #buffer满则调用聚类凝聚新类
            self.DBSCAN(self.buffer)
            self.buffer= []
            # print("buffer empty")
        # -------------
        #  detect start
        #--------------
        A = np.array(x).astype(np.float)
        count_dict = {}
        for key in self.NCDlist.keys():#这里的是实例本身
            count = 0
            if len(self.NCDlist[key]) < 2:#稀疏之后
                continue
            #得到核心点列表
            CPC = np.array(self.NCDlist[key]).astype(np.float)
            CPC = np.row_stack((CPC,A))
            # np.append(CPC, values=A, axis=0)
            #计算距离矩阵
            dist = self.compute_squared_EDM(CPC)
            #遍历最后一列或最后一行，为该点到核心点的距离
            for distance in dist[-1]:
                if distance <= self.epsilon:
                    count+=1
            count_dict[key] = count
        if len(count_dict) == 0:
            #若检测后没有一个核心点接近则认为是新类
            self.buffer.append(x)
            return -5
        MaxKey = max(count_dict, key = count_dict.get)     #找到邻域内核心点最多的类
        Maxvalue = count_dict[MaxKey]
        if Maxvalue > self.minPts : #邻域内核心点数量满足则为正常类,
            self.NCDlist[MaxKey].append(A)#更新已知类的检测器信息
            self.minPts = self.beta * len(self.NCDlist[MaxKey]) # minPts越小，E as N越小
            return MaxKey
        else:
            #若检测后核心点数量不够则认为是新类
            self.buffer.append(x)
            return -5
    #检测出新类时更新检测器
    def updateNCD(self,coreIns):
        #将新检测出来的类，添加到NCD中,检测出来不知道是正类还是反类，只是凝聚了？
        #每个类都有一个
        #NCDlist的元素不是同一个类型，前几个是list后头的是array
        i = coreIns[0][-1]
        # -------------
        # update CONFUSION_MATRIX based on the buffer
        # -------------
        for ins in coreIns:
            self.update_confusion_matrix(ins[-1], i)
        if i in self.clusterList and i in self.NCDlist.keys():#如果该类之前已经被凝聚，则直接取代，或者添加？,之前凝聚过
            self.NCDlist[i] = self.NCDlist[i]+coreIns#说明E被定为N，说明e太小
            # self.NCDlist[i] = coreIns# 直接取代？ 时间未显著提高，准确率接近
            # print("recluster")
        #某类已经被凝聚之后又被凝聚，说明过于敏感e需要变大
            #special for IMDB to limit
            if self.alpha < 1: #normal is 1 Set is used to limit and IMDB is 0.7
                self.alpha += self.e_step
                # print("update alpha"+str(self.alpha))
            # for IMDB
            # else:
            #     self.alpha = 0.65
                # print("reset alpha")
            # if self.alpha < 1: #normal is 1 Set is used to limit
            #     self.alpha += self.e_step
            #     print("update alpha"+str(self.alpha))
            disMat = self.compute_squared_EDM(self.NCDlist[i])
            self.epsilon = self.alpha * np.mean(disMat)#当有同样的类被凝聚出现时，说明e太小了，e需要变大，否则维持不变,扩充minPts也是必须的，需要更多的Minpts


            if len(self.NCDlist[i]) > self.window:#对核心点列表稀疏,大于buffer则稀疏
                #距离稀疏，可能导致列表内仅剩一个或没有核心点，造成计算距离无法通过
                #筛选距离核心点，剔除多余点
                core_points_index = np.where(np.sum(np.where(disMat <= self.epsilon, 1, 0), axis=1) >= self.minPts)[0]
                tmp = [self.NCDlist[i][j] for j in core_points_index]
                if len(tmp) > self.window:
                    tmp = sample(self.NCDlist[i], int(self.sparse * self.window))#直接随机筛除
                # if len(tmp) > int(self.minPts):
                #     tmp = sample(self.NCDlist[i], int(self.sparse * self.minPts))#直接随机筛除
                self.NCDlist[i] = tmp
            # print(str(i)+": the list is rebuild, epsilon growing... ")
        else:
            self.upClass(i) #将检测到的新类返回给分类器，同时保存在检测器中(need only one to get the corresponding label )
            self.NCDlist[i] = coreIns
            print("get new class")
    #计算欧氏距离
    def compute_squared_EDM(self, X):
        #squareform函数，输入是稀疏的就返回原本的，输入非稀疏的就返回稀疏的
        #pdist计算各行向量之间的距离，返回值是一个长为n(n-1)/2的行向量
        #squareform将pdist的返回值重新变成一个nxn的对称矩阵

        X = np.array(X, dtype = np.float)
        return squareform(pdist(X,metric='euclidean'))#返回一个稀疏矩阵，存储矩阵的上三角，为各点的距离,输入值X应该为ndarray

    def upClass(self, newCLass):
        # i = len(self.clusterList)#错误的检测出新类，将旧类凝聚成新的类
        # if i < len(self.CLASSES) and str(newCLass) not in self.clusterList:
        # self.clusterList.append(str(newCLass))
        self.clusterList.append(newCLass)  #for IMDB 识别出来的类别与已获得类别重复：数据类型不同
        print("get new class")
        print(self.clusterList , self.CLASSES)

    def DBSCAN(self, data):# 这里的data是buffer，补全之后
        # 获得距离矩阵，这里是计算各点之间的距离直接计算了各个点之间的距离

        dataset = np.delete(data, -1, axis=1)
        disMat = self.compute_squared_EDM(dataset)#这里存储的各个点之间的距离
        # if self.epsilon == 0:
        #根据距离矩阵确定e
        self.epsilon  =  self.alpha * np.mean(disMat)# 每次聚类都重新对self.e赋值
        self.minPts = self.beta * np.size(disMat[0],0)  #IMDB的时Min太大
        # 获得数据的行和列(一共有n条数据)
        n = len(data)
        # 将矩阵的中小于eps的数赋予1，大于eps的数赋予零，然后1代表对每一行求和,然后求核心点坐标的索引
        core_points_index = np.where(np.sum(np.where(disMat <= self.epsilon, 1, 0), axis=1) >= self.minPts)[0]#计算出邻域内点的个数
        # 初始化类别，-1代表未分类。
        labels = np.full((n,), -1) #构建一个数组，用-1作为填充值，预填充，-1代表未被访问，有n个实例，因此label的size是n，这里并不是真实标签，只是用来标记这些点是否被访问
        clusterId = 0 #标记点是否被访问
        returnSeed = []
        # 遍历所有的点
        for pointId in core_points_index:
            # 如果点未被分类，将其作为的种子点，开始寻找相应簇集
            if (labels[pointId] == -1):
                newClass = []
                # 首先将点pointId标记为当前类别(即标识为已操作)
                labels[pointId] = clusterId#标记该点被访问，也可以用类标签对应
                # 然后寻找种子点的eps邻域且没有被分类的点，将其放入种子集合
                neighbour = np.where((disMat[:, pointId] <= self.epsilon) & (labels==-1))[0]
                seeds = neighbour.tolist()
                # 通过种子点，开始生长，寻找密度可达的数据点，一直到种子集合为空，一个簇集寻找完毕
                while len(seeds) > 0:
                    # 弹出一个新种子点
                    newPoint = seeds.pop()
                    # 将newPoint标记为当前类,因为这个种子点是当前簇所生成的点，必然属于当前类，被访问（同时也能标记类别）
                    labels[newPoint] = clusterId
                    # 寻找newPoint种子点eps邻域（包含自己）
                    queryResults = np.where(disMat[:,newPoint] <= self.epsilon)[0]
                    # 如果newPoint属于核心点，那么newPoint是可以扩展的，即密度是可以通过newPoint继续密度可达的
                    if len(queryResults) >= self.minPts:
                        # 将邻域内且没有被分类的点压入种子集合
                        newClass.append(newPoint) #该点邻域内个数大于MinPts,该点属于对应新类，将该点加入newClass
                        for resultPoint in queryResults:
                            if labels[resultPoint] == -1:
                                seeds.append(resultPoint)
                # 簇集生长完毕(此时仍无法判断该类是否存在)
                returnSeed.append(list(set(newClass)))#将新类的核心点列表索引存储
                # 将簇加入类别标签,并初始化该类的分类器
                # # 对于只有一个新类，这里可以不用，标记类别
                # self.clusterId = self.clusterId + 1
        #NCD核心点列表添加类别的核心点(应该是实例本身，而不是索引),通过核心点索引，选出核心点本身
        for newClass in returnSeed:
            core_ins = [ins for ins in data if data.index(ins) in newClass]#这里返回的，导致没更新
            if len(core_ins) > 0:
                self.updateNCD(core_ins)#
            else:
                #此处需要min小，或e变大
                print("can not cluster new class")

                if self.alpha < 1:
                    self.alpha += self.e_step
                    print("update alpha:"+str(self.alpha))
                #specical for IMDB
                # else:
                #     self.alpha = 0.65
                #     print("reset alpha")
                #for nomal
                # if self.alpha < 1: #normal is 1 Set is used to limit
            #     self.alpha += self.e_step
            #     print("update alpha"+str(self.alpha))
        #是否应该返回核心点，利用核心点更新分类器时需要使用，

    def test_newClass(self,real_class):
        if real_class not in self.clusterList:#class是数据集本来有多少类，cluster是凝聚出来有多少类
            return True
        else:
            return False


    def __initialize_weights(self):
        self.rFeatures(self.fremove)#"variable"
        # self.rDataTrapezoidal(self.features_left)#"Trapezoidal"
        for c in self.CLASSES:
            self.WEIGHTS[c] = OrderedDict()
            for a in self.ATTRIBUTES:
                if self.x_attributes[a.NAME] == 1:#只初始化，特征存在的相应权重
                    self.WEIGHTS[c][a.NAME] = 0.2 * random.random() - 0.1
        self.stability = OrderedDict()
        self.A = dict()
        self.A_= dict()
        self.keyCount = dict() #权重策略，初始化为1
        self.count = dict() #计数，统计出现多少次
        self.R = 0.00001



    def test(self, instance):

        if self._IS_READY:
            y = instance[-1]
            X = list(self.rFeatures(self.fremove, instance[0:len(instance)-1]))
            X.append(y)
            y_predicted = self.NCD(X)
            if y_predicted == -5:#如果检测出来新类不更新混淆矩阵
                return y_predicted
            elif y_predicted == -3:
                return y_predicted
            else:
                self.update_confusion_matrix(y, y_predicted)
                return y_predicted
        else:
            print("Please train a DyfAndNCD classifier first!")
            exit()

    def reset(self):
        self.Is_Drift = True
        print("Dyf need do nothing")

    def expand_space(self, X):#补全特征空间,并且将实例和分类器扩充到全局特征空间
        self.n_keys = dict() #定义新特征
        self.e_keys = dict()
        # self.s_keys = dict()

        self.e_weights = self.WEIGHTS#每个类都有一个weight，但是feature space是相同的
        for key in findDifferentKeys(X, self.WEIGHTS[list(self.CLASSES)[0]]):
            for c in list(self.CLASSES):
                self.WEIGHTS[c][key] = 0
            self.n_keys[key] = 1
        for key in findDifferentKeys(self.WEIGHTS[list(self.CLASSES)[0]],X):
            X[key] = 0
            self.e_keys[key] = 1
        X["bias"] = 1
        return X

    def update_stability(self, X):# feature space 扩充完毕，直接输入扩充过后的实例,# 方差比例越大信息量越大,出现为1，不出现为零，统计离散度
        self.count += 1
        #初始化信息权重
        if self.count == 1:
            self.A_ = X
            self.A = X
            for i in range(0,len(X)):
                self.stability.append(0.0000001)
            self.feature_count = self.stability
        else:
            for ins in range(0, len(X)):
                self.stability[ins] = (self.count-1)/self.count**2*(X[ins]-self.A[ins])**2+(self.count-1)/self.count*self.stability[ins]

            sta_sum = sum(self.stability)

            for ins in range(0, len(self.stability)):
                self.feature_count[ins] = self.stability[ins]/sta_sum

    def upEKeys(self): #应对飘逸策略依旧可以用
        for key in self.e_keys:
            # print(getStability(key))
            # self.stability[key] = self.stability[key]/(1-self.alpha)
            self.stability[key] = self.alpha * self.stability[key]

    def upper_bound(self,x,weights):
        # x=list(x)
        # w=list(self.weights)
        x_norm = math.sqrt(np.dot(x,x))
        w_norm = math.sqrt(dotDict(weights,weights))
        # gamma  = self.min_gamma
        if x_norm > self.R:
            self.R = x_norm
        theta = self.R * w_norm / self.gamma
        if theta == 0:
            theta=0.1
        return theta


