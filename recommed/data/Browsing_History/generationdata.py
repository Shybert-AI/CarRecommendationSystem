# -*- coding:utf-8 -*-
"""
假设数据来源与数据库
总共1000个用户，生成100000条用户的浏览记录，假设所有汽车库存数大于1
"""
import pandas as pd
import os
import sys
import time
import random

#sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(),"..","data\pre_data")))
#os.chdir(os.path.abspath(os.path.join(os.getcwd(),"..","data\pre_data")))

path = os.path.join(os.getcwd(),"..\..\..\data\pre_data")

data1 = pd.read_csv(os.path.join(path,"pre_used_car_testA.csv"))
data2 = pd.read_csv(os.path.join(path,"pre_used_car_testB.csv"))
data3 = pd.read_csv(os.path.join(path,"pre_used_car_train.csv"))
data = pd.concat([data1,data2,data3],axis=0)
data = data.reset_index(drop=True)  #重置索引
#用户集合
user_set = list(range(1,1001))
#汽车集合
car_set = set(data["name"].values.tolist())

#假定浏览时间为2020-01-01-00-00-00至2020-12-30-11-59-59
# 字符类型的时间
start_time = '2020-01-01 00:00:00'
end_time = '2020-12-30 23:59:59'
# 转为时间数组
timeArray = time.strptime(start_time, "%Y-%m-%d %H:%M:%S")    
# 转为时间戳
timeStamp1 = int(time.mktime(timeArray))
timeStamp2 = int(time.mktime(time.strptime(end_time, "%Y-%m-%d %H:%M:%S")))
#时间戳集合
time_set = list(range(timeStamp1,timeStamp2+1))

#时间戳转换成时间格式
#timeStamp = 1609343999
#timeArray = time.localtime(timeStamp)
#otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
#print(otherStyleTime)   # '2020-12-30 23:59:59'
print(f"user={len(user_set)},item ={len(car_set)},汽车的浏览时间为：{start_time}值{end_time}")
records = pd.DataFrame([{"user_id":0,"item_id":1,"time":2}])
records.drop(index=0,inplace=True)

t = 100000
#随机生成浏览记录
time1 = time.time()
for i in range(t):
    record = pd.DataFrame([{"user_id":random.sample(user_set,1)[0],"item_id":random.sample(car_set,1)[0],"time":random.sample(time_set,1)[0]}])
    records = pd.concat([records,record]) #拼接
#按浏览时间排序    
records.sort_values(by="time", inplace=True,ascending=True)
#重置索引
records.reset_index(drop=True,inplace=True) 
#保存数据
#os.chdir(os.path.abspath(os.path.join(os.getcwd(),"..\..","recommed\data")))
records.to_csv(r"records.csv",index=0)
print(f"总共耗时：totel={round((time.time()-time1)/60,2)}min")


