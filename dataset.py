# -*- coding:utf-8 -*-

"""
由数据分析得到：
1.通过可视化缺失值，['model', 'bodyType', 'fuelType', 'gearbox']列的缺失值为 [1, 4506, 5981, 8680]，考虑含义，进行随机填充
2.通过可视化数据类型，"notRepairedDamage"列中存在24324个"-"值，考虑用nan进行替换 
3.通过可视化特征分布，offerType和seller类别特征严重倾斜，一般不会对预测有什么帮助，因此先删掉 
4.通过可视化特征分布，预测价格是长尾分布，不是服从正态分布，因此采用对数变换
"""
import os
import pandas as pd 
import numpy as np
import random
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
pd.set_option('display.max_columns', None)    #显示所有行数据

train_data = pd.read_csv(r"data\used_car_train.csv",sep=" ")
testA_data = pd.read_csv(r"data\used_car_testA.csv",sep=" ")
testB_data = pd.read_csv(r"data\used_car_testB.csv",sep=" ")
#打印数据的列明
print(train_data.columns.tolist()) #['SaleID', 'name', 'regDate', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'power', 'kilometer', 'notRepairedDamage', 'regionCode', 'seller', 'offerType', 'creatDate', 'price', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']
print(testA_data.columns.tolist()) 

#train_data["notRepairedDamage"].value_counts() 验证缺失值
#考虑到notRepairedDamage为汽车有尚未修复的损坏：是：0，否：1，随机填充
train_data["notRepairedDamage"] = train_data["notRepairedDamage"].apply(lambda x: random.sample(range(2), 1)[0] if x == "-" else x)
testA_data["notRepairedDamage"] = train_data["notRepairedDamage"].apply(lambda x: random.sample(range(2), 1)[0] if x == "-" else x)
testB_data["notRepairedDamage"] = train_data["notRepairedDamage"].apply(lambda x: random.sample(range(2), 1)[0] if x == "-" else x)
#train_data["notRepairedDamage"].value_counts() 验证缺失值

#"bodyType", "fuelType", "gearbox"的含义为：采用随机填充
#bodyType	车身类型：豪华轿车：0，微型车：1，厢型车：2，大巴车：3，敞篷车：4，双门汽车：5，商务车：6，搅拌车：7
#fuelType	燃油类型：汽油：0，柴油：1，液化石油气：2，天然气：3，混合动力：4，其他：5，电动：6
#gearbox	变速箱：手动：0，自动：1
#用小数点判断必须将数据转换成浮点型
train_data.astype(float)
train_data["fuelType"] = train_data["bodyType"].apply(lambda x: random.sample(range(7), 1)[0] if str(x).find(".") == -1 else x)
train_data["gearbox"] = train_data["bodyType"].apply(lambda x: random.sample(range(2), 1)[0] if str(x).find(".") == -1 else x)

testA_data.astype(float)
testA_data["bodyType"] = testA_data["bodyType"].apply(lambda x: random.sample(range(8), 1)[0] if str(x).find(".") == -1 else x)
testA_data["fuelType"] = testA_data["bodyType"].apply(lambda x: random.sample(range(7), 1)[0] if str(x).find(".") == -1 else x)
testA_data["gearbox"] = testA_data["bodyType"].apply(lambda x: random.sample(range(2), 1)[0] if str(x).find(".") == -1 else x)

testB_data.astype(float)
testB_data["bodyType"] = testB_data["bodyType"].apply(lambda x: random.sample(range(8), 1)[0] if str(x).find(".") == -1 else x)
testB_data["fuelType"] = testB_data["bodyType"].apply(lambda x: random.sample(range(7), 1)[0] if str(x).find(".") == -1 else x)
testB_data["gearbox"] = testB_data["bodyType"].apply(lambda x: random.sample(range(2), 1)[0] if str(x).find(".") == -1 else x)
#删除多余列
train_data.drop(columns=["offerType","seller"],inplace=True)
testA_data.drop(columns=["offerType","seller"],inplace=True)
testB_data.drop(columns=["offerType","seller"],inplace=True)
#使用时间
user_time_train = pd.to_datetime(train_data['creatDate'], format='%Y%m%d', errors='coerce') - pd.to_datetime(train_data['regDate'], format='%Y%m%d', errors='coerce')
train_data = pd.concat([train_data,user_time_train],axis=1)
train_data.rename(columns ={0:"user_time"},inplace=True)
train_data["user_time"] = train_data["user_time"].apply(lambda x:x.days)
train_data.drop(columns=['creatDate','regDate'],inplace=True)

user_time_testA = pd.to_datetime(testA_data['creatDate'], format='%Y%m%d', errors='coerce') - pd.to_datetime(testA_data['regDate'], format='%Y%m%d', errors='coerce')
testA_data = pd.concat([testA_data,user_time_testA],axis=1)
testA_data.rename(columns ={0:"user_time"},inplace=True)
testA_data["user_time"] = testA_data["user_time"].apply(lambda x:x.days)
testA_data.drop(columns=['creatDate','regDate'],inplace=True)

user_time_testB = pd.to_datetime(testB_data['creatDate'], format='%Y%m%d', errors='coerce') - pd.to_datetime(testB_data['regDate'], format='%Y%m%d', errors='coerce')
testB_data = pd.concat([testB_data,user_time_testB],axis=1)
testB_data.rename(columns ={0:"user_time"},inplace=True)
testB_data["user_time"] = testB_data["user_time"].apply(lambda x:x.days)
testB_data.drop(columns=['creatDate','regDate'],inplace=True)
#通过预测值price可视化的数据的标签呈现长尾分布，不利于我们的建模预测。原因是很多模型都假设数据误差项符合正态分布，而长尾分布的数据违背了这一假设

#可视化变换前后
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train_data["price"])
plt.subplot(1,2,2)
sns.distplot(train_data["price"][train_data["price"] < np.quantile(train_data["price"], 0.9)])
plt.savefig("预测值price变换前的图像.png") #保存
#plt.show()  #显示

train_data["newprice"] = np.log(train_data["price"] + 1)
#可视化变换前后

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train_data["newprice"])
plt.subplot(1,2,2)
sns.distplot(train_data["newprice"][train_data["newprice"] < np.quantile(train_data["newprice"], 0.9)])
plt.savefig("预测值price变换后的图像.png")
#plt.show()

#保存处理后的数据
path = "data\pre_data"
if not os.path.isdir(path):
    os.makedirs(path)
train_data.to_csv(os.path.join(path,"pre_used_car_train.csv"),index=0)
testA_data.to_csv(os.path.join(path,"pre_used_car_testA.csv"),index=0)
testB_data.to_csv(os.path.join(path,"pre_used_car_testB.csv"),index=0)
    