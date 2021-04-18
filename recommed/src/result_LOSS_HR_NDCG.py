# -*- coding:utf-8 -*-

import os
from json import load
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)  #显示所有数据
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

path = os.path.join(os.getcwd(),"..","data\RealRecord") #注意路径修改数据集名称
with open(os.path.join(path,"resualts.json")) as f:
    results = load(f)
#with open(os.path.join(path,"sequential_resualts.json")) as f:
    #results = load(f)    
    
dev = np.array(results["dev"])
dev_HR_NDCG = {"HR":[],"NDCG":[]}
for i,j in enumerate(dev):
    dev_HR_NDCG["NDCG"].append(j['NDCG@10'])
    dev_HR_NDCG["HR"].append(j['HR@10'])
    
test = np.array(results["test"])
test_HR_NDCG = {"HR":[],"NDCG":[]}
for i,j in enumerate(test):
    test_HR_NDCG["NDCG"].append(j['NDCG@10'])
    test_HR_NDCG["HR"].append(j['HR@10'])

plt.subplot(2,1,1)
plt.title("RESULT_LOSS_HR命中率")
plt.plot(np.round(np.array(results["loss"]),3),label="loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.subplot(2,1,2)    
plt.plot(np.round(np.array(test_HR_NDCG["HR"])*100,3),label="test_HR",c="r")
plt.plot(np.round(np.array(dev_HR_NDCG["HR"])*100,3),label="dev_HR")
plt.xlabel("epoch")
plt.ylabel("HR%")
plt.legend()
plt.savefig(os.path.join(path,"结果可视化.png"))   #保存图像
plt.show()   #显示图像
