# -*- coding:utf-8 -*-

import pandas as pd
pd.set_option('display.max_columns', None)  #显示所有数据
from sklearn.decomposition import PCA
#PCA https://blog.csdn.net/sxb0841901116/article/details/83816356
from matplotlib import pyplot as plt
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance,plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV  #调参
import logging
import sys
import os
import datetime
import warnings
from joblib import dump,load
warnings.filterwarnings('ignore')
# 解决中文和负号显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 使显示图标自适应
plt.rcParams['figure.autolayout'] = True

log_path = os.makedirs(r".\log") if not os.path.exists(r".\log")  else  1  #创建.\log目录
logging.basicConfig(filename=r".\log\root.log", level=20,format="%(asctime)s %(message)s",datefmt="%Y-%m-%d %I:%M:%S %p")
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info('-' * 45 + ' BEGIN: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' + '-' * 45)
#数据集(数据已经预处理过)

train = pd.read_csv(r"../data/pre_data/pre_used_car_train.csv",sep=",")
logging.info(f"train.shape={train.shape},test.shape={test.shape}")

for i in train.columns:  
    train[i] = train[i].apply(lambda x: float(x))          #将字符转化为数值

#数据集划分 
x = train.drop(columns=["newprice","price"]).values.tolist()
y = train["newprice"].values.tolist()
X_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2021)
X_train,x_test,y_train,y_test = np.array(X_train),np.array(x_test),np.array(y_train),np.array(y_test)

#调参数
#logging.info('-' * 45 + ' DEBUG PARAMETER: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' + '-' * 45)
#logging.info('-' * 20 +"调参："+'-' * 20)
###模型训练
#model = XGBRegressor(booster="gbtree",      
                         #objective='reg:gamma',
                         #)
#model = model.fit(X_train, y_train)


## 模型预测
#y_pred = model.predict(x_test)
#loss = mean_squared_error(y_test, y_pred)      
#logging.info(f"误差loss={loss}时,模型的参数为：")  #loss=0.0602743564578782
#logging.info(model)    #找出初始模型的参数

##调参1，将上述参数代入进行调参
##-------------------------------------------------------------------------------
#logging.info('-' * 20 +"调参1："+'-' * 20)
#model2 = XGBRegressor(base_score=0.5,
                      #booster='gbtree',  #["gbtree","gbliner"]
                      #colsample_bylevel=1,
                      #colsample_bynode=1, 
                      #colsample_bytree=0.8, #控制每棵随机采样的列数的占比,0.5-1
                      #gamma=0,  #Gamma指定了节点分裂所需的最小损失函数下降值,需要调整
                      #gpu_id=-1,
                      #importance_type='gain',
                      #interaction_constraints='',
                      #learning_rate=0.300000012, #学习率
                      #max_delta_step=0, #限制每棵树权重改变的最大步长，通常，这个参数不需要设置，如果它被赋予了某个正值，那么它会让这个算法更加保守
                      #max_depth=6,   #树的最大深度。 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本,需要调整
                      #min_child_weight=1,     #当它的值较大时，可以避免模型学习到局部的特殊样本。 但是如果这个值过高，会导致欠拟合,需要调整
                      #monotone_constraints='()',
                      #n_estimators=100, #迭代次数
                      #n_jobs=0, 
                      #num_parallel_tree=1,
                      #objective='reg:gamma',
                      #random_state=2021,
                      #reg_alpha=0, #L1正则化,减轻过拟合
                      #reg_lambda=1,#L2正则化,减轻过拟合
                      #scale_pos_weight=None, #在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛
                      #subsample=0.8,  #控制对于每棵树，随机采样的比例,典型值：0.5-1
                      #tree_method='exact',
                      #validate_parameters=1,
                      #eval_metric="rmse",
                      #verbosity=None) #训练过程中打印的日志等级，0 (silent), 1 (warning), 2 (info), 3 (debug)


#parameters1 = {'max_depth': list(range(3,10)),"min_child_weight":list(range(1,6))}
#model = model2
#clf = GridSearchCV(model, parameters1, cv=5)
#clf = clf.fit(X_train, y_train)
#logging.info(clf)

## 模型预测
#y_pred1 = clf.predict(x_test)
#loss2 = mean_squared_error(y_test, y_pred1)      
#logging.info(f"误差loss={loss2}时,模型的参数为：")  #loss=0.06039603576399363
#logging.info(clf.best_params_)    #{'max_depth': 7, 'min_child_weight': 5}
#-------------------------------------------------------------------------------------


#调参2
#-------------------------------------------------------------------------------------
#logging.info('-' * 20 +"调参2："+'-' * 20)
#model2 = XGBRegressor(base_score=0.5,
                      #booster='gbtree',  #["gbtree","gbliner"]
                      #colsample_bylevel=1,
                      #colsample_bynode=1, 
                      #colsample_bytree=0.8, #控制每棵随机采样的列数的占比,0.5-1
                      #gamma=0,  #Gamma指定了节点分裂所需的最小损失函数下降值,需要调整
                      #gpu_id=-1,
                      #importance_type='gain',
                      #interaction_constraints='',
                      #learning_rate=0.300000012, #学习率
                      #max_delta_step=0, #限制每棵树权重改变的最大步长，通常，这个参数不需要设置，如果它被赋予了某个正值，那么它会让这个算法更加保守
                      #max_depth=6,   #树的最大深度。 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本,需要调整
                      #min_child_weight=1,     #当它的值较大时，可以避免模型学习到局部的特殊样本。 但是如果这个值过高，会导致欠拟合,需要调整
                      #monotone_constraints='()',
                      #n_estimators=100, #迭代次数
                      #n_jobs=0, 
                      #num_parallel_tree=1,
                      #objective='reg:gamma',
                      #random_state=2021,
                      #reg_alpha=0, #L1正则化,减轻过拟合
                      #reg_lambda=1,#L2正则化,减轻过拟合
                      #scale_pos_weight=None, #在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛
                      #subsample=0.8,  #控制对于每棵树，随机采样的比例,典型值：0.5-1
                      #tree_method='exact',
                      #validate_parameters=1,
                      #eval_metric="rmse",
                      #verbosity=None) #训练过程中打印的日志等级，0 (silent), 1 (warning), 2 (info), 3 (debug)



#parameters2 = {'learning_rate': [0.01,0.05,0.1,0.2,0.3,0.4,0.5],"n_estimators":list(range(10,150,5))}
#model = model2
#clf = GridSearchCV(model, parameters2, cv=5)
#clf = clf.fit(X_train, y_train)
#logging.info(clf)
#logging.info(clf.best_params_)

## 模型预测
#y_pred1 = clf.predict(x_test)
#loss2 = mean_squared_error(y_test, y_pred1)      
#logging.info(f"误差loss={loss2}时,模型的参数为：")  #loss=0.05778166164973256
##logging.info(clf.best_params_)    #{'learning_rate': 0.2, 'n_estimators': 145}
#-------------------------------------------------------------------------------------

#调参3
#-------------------------------------------------------------------------------------
#logging.info('-' * 20 +"调参3："+'-' * 20)
#model2 = XGBRegressor(base_score=0.5,
                      #booster='gbtree',  #["gbtree","gbliner"]
                      #colsample_bylevel=1,
                      #colsample_bynode=1, 
                      #colsample_bytree=0.8, #控制每棵随机采样的列数的占比,0.5-1
                      #gamma=0,  #Gamma指定了节点分裂所需的最小损失函数下降值,需要调整
                      #gpu_id=-1,
                      #importance_type='gain',
                      #interaction_constraints='',
                      #learning_rate=0.2, #学习率
                      #max_delta_step=0, #限制每棵树权重改变的最大步长，通常，这个参数不需要设置，如果它被赋予了某个正值，那么它会让这个算法更加保守
                      #max_depth=6,   #树的最大深度。 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本,需要调整
                      #min_child_weight=1,     #当它的值较大时，可以避免模型学习到局部的特殊样本。 但是如果这个值过高，会导致欠拟合,需要调整
                      #monotone_constraints='()',
                      #n_estimators=145, #迭代次数
                      #n_jobs=0, 
                      #num_parallel_tree=1,
                      #objective='reg:gamma',
                      #random_state=2021,
                      #reg_alpha=0, #L1正则化,减轻过拟合
                      #reg_lambda=1,#L2正则化,减轻过拟合
                      #scale_pos_weight=None, #在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛
                      #subsample=0.8,  #控制对于每棵树，随机采样的比例,典型值：0.5-1
                      #tree_method='exact',
                      #validate_parameters=1,
                      #eval_metric="rmse",
                      #verbosity=None) #训练过程中打印的日志等级，0 (silent), 1 (warning), 2 (info), 3 (debug)



#parameters3 = {'subsample': [0.5,0.6,0.7,0.8,0.85,0.9,0.95,1],"colsample_bytree":[0.5,0.6,0.7,0.8,0.85,0.9,0.95,1]}
#model = model2
#clf = GridSearchCV(model, parameters3, cv=5)
#clf = clf.fit(X_train, y_train)
#logging.info(clf)

## 模型预测
#y_pred1 = clf.predict(x_test)
#loss2 = mean_squared_error(y_test, y_pred1)      
#logging.info(f"误差loss={loss2}时,模型的参数为：")  #loss=0.05882159937661019
#logging.info(clf.best_params_)     #{'colsample_bytree': 0.5, 'subsample': 0.95}
#调参4
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#调参5
#-------------------------------------------------------------------------------
#logging.info('-' * 20 +"调参5："+'-' * 20)
#model2 = XGBRegressor(base_score=0.5,
                      #booster='gbtree',  #["gbtree","gbliner"]
                      #colsample_bylevel=1,
                      #colsample_bynode=1, 
                      #colsample_bytree=0.5, #控制每棵随机采样的列数的占比,0.5-1
                      #gamma=0,  #Gamma指定了节点分裂所需的最小损失函数下降值,需要调整
                      #gpu_id=-1,
                      #importance_type='gain',
                      #interaction_constraints='',
                      #learning_rate=0.2, #学习率
                      #max_delta_step=0, #限制每棵树权重改变的最大步长，通常，这个参数不需要设置，如果它被赋予了某个正值，那么它会让这个算法更加保守
                      #max_depth=7,   #树的最大深度。 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本,需要调整
                      #min_child_weight=5,     #当它的值较大时，可以避免模型学习到局部的特殊样本。 但是如果这个值过高，会导致欠拟合,需要调整
                      #monotone_constraints='()',
                      #n_estimators=200, #迭代次数
                      #n_jobs=0, 
                      #num_parallel_tree=1,
                      #objective='reg:gamma',
                      #random_state=2021,
                      #reg_alpha=0, #L1正则化,减轻过拟合
                      #reg_lambda=1,#L2正则化,减轻过拟合
                      #scale_pos_weight=None, #在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛
                      #subsample=0.95,  #控制对于每棵树，随机采样的比例,典型值：0.5-1
                      #tree_method='exact',
                      #validate_parameters=1,
                      #eval_metric="rmse",
                      #verbosity=None) #训练过程中打印的日志等级，0 (silent), 1 (warning), 2 (info), 3 (debug)

#parameters4 = {'reg_alpha': [0.1,0.3,0.5,0.6,0.7,0.8,0.85,0.9,0.95,1],"reg_lambda":[0.1,0.3,0.5,0.6,0.7,0.8,0.85,0.9,0.95,1]}
#model = model2
#clf = GridSearchCV(model, parameters4, cv=5)
#clf = clf.fit(X_train, y_train)
#logging.info(clf)

## 模型预测
#y_pred1 = clf.predict(x_test)
#loss2 = mean_squared_error(y_test, y_pred1)      
#logging.info(f"误差loss={loss2}时,模型的参数为：")  #loss=0.056469999954277046
#logging.info(clf.best_params_)     #{'reg_alpha': 0.85, 'reg_lambda': 0.95}
#-------------------------------------------------------------------------------------
#最终模型
#-------------------------------------------------------------------------------
#logging.info('-' * 20 +"最终模型："+'-' * 20)
model = XGBRegressor( base_score=0.5,
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
features_name = train.columns.tolist()
features_name.remove("price")
features_name.remove("newprice")
model = model.fit(X_train,y_train)

# 模型预测
y_pred = model.predict(x_test)
loss2 = mean_squared_error(y_test, y_pred)      
logging.info(f"误差loss={loss2}时,模型的参数为：")    #loss=0.056469999954277046
#保存模型
dump(model, "model.pkl")   
#可视化
plt.plot(y_pred[:200],c="r",label="预测价格")
plt.plot(y_test[:200],c="b",label="真实价格")
plt.legend()
plt.savefig("预测价格和真实价格.png")   #保存图片
#plt.show()   #显示图片
#-------------------------------------------------------------------------------------

#分析特征重要性排序和可视化树的生成情况
features_name = train.columns.tolist()
features_name.remove("price")
features_name.remove("newprice")
## 显示重要特征
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features_name)
dtest = xgb.DMatrix(x_test)

param = {"base_score":0.5,"booster":'gbtree',"colsample_bylevel":1,"colsample_bynode":1,"colsample_bytree":0.5,
         "gamma":0,"gpu_id":-1,"importance_type":'gain',"learning_rate":0.2,
         "max_delta_step":0,"max_depth":7,"min_child_weight":5,"n_estimators":200,"num_parallel_tree":1,"objective":'reg:gamma',
         "random_state":2021,"reg_alpha":0.85,"reg_lambda":0.95,"subsample":0.95,"tree_method":'exact',"validate_parameters":1}

model = xgb.train(param, dtrain)

#特征重要性排序
plt.rcParams["figure.figsize"] = (16, 16)
plot_importance(model, title='特征重要性排序', xlabel='得分', ylabel='特征', grid=False)
plt.savefig("feature.png")
#plt.show()

# 可视化树的生成情况，num_trees是树的索引
plt.rcParams["figure.figsize"] = (16, 16)
plot_tree(model, num_trees=5) 
plt.savefig("trees.png")
#plt.show()

logging.info('-' * 45 + ' END: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' + '-' * 45)