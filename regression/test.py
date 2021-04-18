# -*- coding:utf-8 -*-
from  joblib import load
import pandas as pd
import numpy as np
from functools import reduce
import math
import numpy as np

model = load("last_model.pkl")
test = pd.read_csv(r"../data/pre_data/pre_used_car_testA.csv",sep=",")
for i in test.columns:  
    test[i] = test[i].apply(lambda x: float(x))          #将字符转化为数值
test = test.values.tolist()
test = np.array(test)
print(test[0].shape) #(27,)
predict = model.predict(test)  #数据预处理时对价格求取了对数，因此需要将预测出的值取指数运算
print(predict[:3])  #[10.443367   5.7599545  8.697652 ]

#result_predict = list(map(lambda x:math.exp(x),predict)) #
#print(result_predict[:3]) [34316.00026429069, 317.33387482866397, 5988.8330938887]
result_predict =  np.round(np.exp(np.array(predict)),2) 
print(list(result_predict)[:3])  #[34316.0, 317.33, 5988.83]