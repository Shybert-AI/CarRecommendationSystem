# -*- coding:utf-8 -*-
"""
对比两组实验：
1.观测预测价格经过log变换和没经过变换，最终的损失
2.观测经过主成分分析和原始数据预测，最终的损失
"""

import pandas as pd
pd.set_option('display.max_columns', None)  #显示所有数据
from sklearn.decomposition import PCA
#PCA https://blog.csdn.net/sxb0841901116/article/details/83816356
from matplotlib import pyplot as plt
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
from joblib import dump
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
# 解决中文和负号显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

train = pd.read_csv(r"../data/pre_data/pre_used_car_train.csv",sep=",")
for i in train.columns:  
    train[i] = train[i].apply(lambda x: float(x))          #将字符转化为数值
    train[i] = train[i].apply(lambda x: 0 if str(x).find(".") == -1 else x)

#变换前的价格   
x = train.drop(columns=["newprice","price"]).values.tolist()
y1 = train["newprice"].values.tolist()  #变换后的价格
X_train_1,x_test_1,y_train_1,y_test_1 = train_test_split(x,y1,test_size=0.2,random_state=2021,shuffle=False)
X_train_1,x_test_1,y_train_1,y_test_1 = np.array(X_train_1),np.array(x_test_1),np.array(y_train_1),np.array(y_test_1)
#变换后的价格
y2 = train["price"].values.tolist()     #变换前的价格
X_train_2,x_test_2,y_train_2,y_test_2 = train_test_split(x,y2,test_size=0.2,random_state=2021,shuffle=False)
X_train_2,x_test_2,y_train_2,y_test_2 = np.array(X_train_2),np.array(x_test_2),np.array(y_train_2),np.array(y_test_2)
#PCA 主成分分析
X_train_3,x_test_3,y_train_3,y_test_3 = train_test_split(x,y1,test_size=0.2,random_state=2021,shuffle=False)
X_train_3,x_test_3,y_train_3,y_test_3 = np.array(X_train_3),np.array(x_test_3),np.array(y_train_3),np.array(y_test_3)

model1 = XGBRegressor( base_score=0.5,
                      booster='gbtree',  #["gbtree","gbliner"]
                      colsample_bylevel=1,
                      colsample_bynode=1, 
                      colsample_bytree=0.5, #控制每棵随机采样的列数的占比,0.5-1
                      gamma=0,  #Gamma指定了节点分裂所需的最小损失函数下降值,需要调整
                      gpu_id=-1,
                      importance_type='gain',
                      interaction_constraints='',
                      learning_rate=0.2, #学习率
                      max_delta_step=0, #限制每棵树权重改变的最大步长，通常，这个参数不需要设置，如果它被赋予了某个正值，那么它会让这个算法更加保守
                      max_depth=7,   #树的最大深度。 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本,需要调整
                      min_child_weight=5,     #当它的值较大时，可以避免模型学习到局部的特殊样本。 但是如果这个值过高，会导致欠拟合,需要调整
                      monotone_constraints='()',
                      n_estimators=200, #迭代次数
                      n_jobs=0, 
                      num_parallel_tree=1,
                      objective='reg:gamma',
                      random_state=2021,
                      reg_alpha=0.85, #L1正则化,减轻过拟合
                      reg_lambda=0.95,#L2正则化,减轻过拟合
                      scale_pos_weight=None, #在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛
                      subsample=0.95,  #控制对于每棵树，随机采样的比例,典型值：0.5-1
                      tree_method='exact',
                      validate_parameters=1,
                      eval_metric="rmse",
                      verbosity=None) #训练过程中打印的日志等级，0 (silent), 1 (warning), 2 (info), 3 (debug)

model2 = XGBRegressor( base_score=0.5,
                      booster='gbtree',  #["gbtree","gbliner"]
                      colsample_bylevel=1,
                      colsample_bynode=1, 
                      colsample_bytree=0.5, #控制每棵随机采样的列数的占比,0.5-1
                      gamma=0,  #Gamma指定了节点分裂所需的最小损失函数下降值,需要调整
                      gpu_id=-1,
                      importance_type='gain',
                      interaction_constraints='',
                      learning_rate=0.2, #学习率
                      max_delta_step=0, #限制每棵树权重改变的最大步长，通常，这个参数不需要设置，如果它被赋予了某个正值，那么它会让这个算法更加保守
                      max_depth=7,   #树的最大深度。 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本,需要调整
                      min_child_weight=5,     #当它的值较大时，可以避免模型学习到局部的特殊样本。 但是如果这个值过高，会导致欠拟合,需要调整
                      monotone_constraints='()',
                      n_estimators=200, #迭代次数
                      n_jobs=0, 
                      num_parallel_tree=1,
                      objective='reg:gamma',
                      random_state=2021,
                      reg_alpha=0.85, #L1正则化,减轻过拟合
                      reg_lambda=0.95,#L2正则化,减轻过拟合
                      scale_pos_weight=None, #在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛
                      subsample=0.95,  #控制对于每棵树，随机采样的比例,典型值：0.5-1
                      tree_method='exact',
                      validate_parameters=1,
                      eval_metric="rmse",
                      verbosity=None) #训练过程中打印的日志等级，0 (silent), 1 (warning), 2 (info), 3 (debug)

model3 = XGBRegressor( base_score=0.5,
                      booster='gbtree',  #["gbtree","gbliner"]
                      colsample_bylevel=1,
                      colsample_bynode=1, 
                      colsample_bytree=0.5, #控制每棵随机采样的列数的占比,0.5-1
                      gamma=0,  #Gamma指定了节点分裂所需的最小损失函数下降值,需要调整
                      gpu_id=-1,
                      importance_type='gain',
                      interaction_constraints='',
                      learning_rate=0.2, #学习率
                      max_delta_step=0, #限制每棵树权重改变的最大步长，通常，这个参数不需要设置，如果它被赋予了某个正值，那么它会让这个算法更加保守
                      max_depth=7,   #树的最大深度。 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本,需要调整
                      min_child_weight=5,     #当它的值较大时，可以避免模型学习到局部的特殊样本。 但是如果这个值过高，会导致欠拟合,需要调整
                      monotone_constraints='()',
                      n_estimators=200, #迭代次数
                      n_jobs=0, 
                      num_parallel_tree=1,
                      objective='reg:gamma',
                      random_state=2021,
                      reg_alpha=0.85, #L1正则化,减轻过拟合
                      reg_lambda=0.95,#L2正则化,减轻过拟合
                      scale_pos_weight=None, #在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛
                      subsample=0.95,  #控制对于每棵树，随机采样的比例,典型值：0.5-1
                      tree_method='exact',
                      validate_parameters=1,
                      eval_metric="rmse",
                      verbosity=None) #训练过程中打印的日志等级，0 (silent), 1 (warning), 2 (info), 3 (debug)

model1 = model1.fit(X_train_1,y_train_1)
model2 = model2.fit(X_train_2,y_train_2)
#PCA降维度
pca = PCA(n_components=6)  #n_components 降低后的维度
pca.fit(X_train_3, y_train_3)
X_train_dunction = pca.transform(X_train_3)
X_test_dunction = pca.transform(x_test_3)
print(X_train_dunction.shape)
print(X_test_dunction.shape)    
model3 = model3.fit(X_train_dunction, y_train_3)

# 模型预测
y_pred_1 = model1.predict(x_test_1)
y_pred_2 = model2.predict(x_test_2)
y_pred_3 = model3.predict(X_test_dunction)
loss_1 = mean_squared_error(y_test_1, y_pred_1)  #price log变换的损失    #loss=0.05646080590937468
print(f"价格price进行log变换的损失为：loss={loss_1}")
loss_2 = mean_squared_error(y_test_2, y_pred_2)  #price log未变换的损失  #loss=1943160.1528287313
print(f"价格price不进行log变换的损失为：loss={loss_2}")
loss_3 = mean_squared_error(y_test_3, y_pred_3)  #PCA特征降维
print(f"特征进行主成分分析PCA的损失为：loss={loss_3}") #PCA特征降维的损失    #loss=0.3917351389225821

#可视化
plt.plot(y_pred_1[:200],c="r",label="预测价格_log变换")
#plt.plot(y_pred_2[:200],c="g",label="预测价格_no_log变换")
plt.plot(y_pred_3[:200],c="y",label="PCA特征降维")
plt.plot(y_test_1[:200],c="b",label="真实价格")
plt.title("预测值和真实值曲线")
plt.legend()
plt.savefig("预测值和真实值曲线.png")   #保存图片
#plt.show()   #显示图片

#分析结论：xgboost对预测值属于长尾分布的需要进行log转换。价格price进行log变换相比使用了主成分分析进行降维，极大减少了预测的不准确性。
#因此最终的模型采用不进行log变化的预测值
dump(model1, "last_model.pkl")
