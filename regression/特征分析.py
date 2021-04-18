# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import types
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#https://blog.csdn.net/Power1_Power2/article/details/79935476


#读取训练数数据集
pf = pd.read_csv("used_car_train.csv",sep=" ")
tf = pd.read_csv("used_car_testA.csv",sep=" ")

#------------数据清洗-------------
pf.drop(columns=["SaleID",'v_0', 'v_1', 'v_2', 'v_3',
       'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12',
       'v_13', 'v_14'], inplace=True)
tf.drop(columns=["SaleID",'v_0', 'v_1', 'v_2', 'v_3',
       'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12',
       'v_13', 'v_14'], inplace=True)
pf.fillna(0,inplace=True) #将NaN填充为0
#pf.dropna(inplace=True)  #将NaN
for i in pf.columns:  
    pf[i] = pf[i].apply(lambda x: 0 if x=="-" else x)    #将-替换成0
    pf[i] = pf[i].apply(lambda x: float(x))          #将字符转化为数值
    tf[i] = pf[i].apply(lambda x: float(x))          #将字符转化为数值
    #pf[i] = pf[i].apply(lambda x: 0 if x=="NaN" else x)  #将NaN替换成0


colormap=plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Correlation between features', y=1.05, size=15)
sns.heatmap(pf,cmap=colormap,linecolor='white',linewidths=0.1,vmax=1.0,square=True,annot=True)
plt.savefig()

print(2333336)