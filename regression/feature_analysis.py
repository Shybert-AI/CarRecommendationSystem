# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.decomposition import PCA
#PCA https://blog.csdn.net/sxb0841901116/article/details/83816356
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

from timeit import timeit

pf = pd.read_csv("used_car_train.csv",sep=" ")
#------------数据清洗-------------
pf.fillna(0,inplace=True) #将NaN填充为0
#pf.dropna(inplace=True)
for i in pf.columns:  
    pf[i] = pf[i].apply(lambda x: 0 if x=="-" else x)    #将-替换成0
    pf[i] = pf[i].apply(lambda x: 0 if x=="NaN" else x)  #将NaN替换成0
    
#X, y = load_digits(return_X_y=True)
X,y = pf.drop(columns=["SaleID",'price']).values,pf['price'].values
print('X shape:', X.shape)  # 此处会看到X是64维的数据
X_train, x_test, y_train, y_test = train_test_split(X, y)
dict_pca = {"PCA":0,"NOPCA":0}


def exec_without_pca():
    #knn_clf = KNeighborsClassifier()
    knn_clf = KNeighborsRegressor()
    print(X_train.shape)
    print(y_train.shape)
    pre = knn_clf.fit(X_train, y_train)
    dict_pca["PCA"]= pre
    print (knn_clf.score(x_test, y_test))
    


def exec_with_pca():
    #knn_clf = KNeighborsClassifier()
    knn_clf = KNeighborsRegressor()
    pca = PCA(n_components=6)  #n_components 降低后的维度
    pca.fit(X_train, y_train)
    X_train_dunction = pca.transform(X_train)
    X_test_dunction = pca.transform(x_test)
    print(X_train_dunction.shape)
    print(X_test_dunction.shape)    
    pre = knn_clf.fit(X_train_dunction, y_train)
    dict_pca["NOPCA"]= pre
    print (knn_clf.score(X_test_dunction, y_test))

def draw_graph():
    
    # 把29维降维1维，进行数据可视化
    pca = PCA(n_components=1)
    pca.fit(X)
    X_reduction = pca.transform(X)
    
    plt.plot(X_reduction[::1000]/X_reduction.max(),c="r",label="PCA")
    plt.plot(y[::1000]/y.max(),c="b",label="NO PCA")
    plt.title("PCA & NO PCA")
    plt.legend()
    plt.savefig("PCA&NOPCA.png") #保存
    #plt.show() #显示

if __name__ == '__main__':
    print ('Time of method[exec_with_pca] costs:',
           timeit('exec_without_pca()', setup='from __main__ import exec_without_pca', number=3))
    print('----' * 10)
    print ('Time of method[exec_with_pca] costs:',
           timeit('exec_with_pca()', setup='from __main__ import exec_with_pca', number=3))
    draw_graph()
    print(243444444)