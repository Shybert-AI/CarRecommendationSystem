# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

def generate_dev_test(data_df,iids,clicked_item_set):   
    uid_dict ={'neg_items':1}
    uid_pfs = pd.DataFrame([uid_dict])
    uid_pfs.drop(index=0,inplace=True)
    data_df.reset_index(drop=True,inplace=True) 
    data_df["neg_items"]=str([0])
    for i, uid in enumerate(data_df['user_id'].values):
        neg_items = random.sample(iids,99)
        user_clicked = clicked_item_set[uid]
        for j in range(len(neg_items)):
            while neg_items[j] in user_clicked:  #生成负样本，负样本不在用户购买的商品中
                neg_items[j] = random.sample(iids,1)[0]
        data_df["neg_items"][i] =neg_items
        
    return  data_df

if __name__ == "__main__":
    out_df = pd.read_csv("records.csv",sep=",")
    clicked_item_set = dict()
    for user_id, seq_df in out_df.groupby('user_id'):   #分组聚合
        clicked_item_set[user_id] = list(set(seq_df['item_id'].values.tolist()))
    iids = sorted(out_df['item_id'].unique())    
    #按8:1:1划分为训练集、验证集、测试集
    out_df = out_df.sample(frac=1)
    train_df = out_df.iloc[:79999,:]
    train_df.sort_values(by="time", inplace=True)
    test_df = out_df.iloc[79999:89999,:]
    dev_df = out_df.iloc[89999:99999,:]
    
    test_df = generate_dev_test(test_df,iids,clicked_item_set)
    dev_df = generate_dev_test(dev_df,iids,clicked_item_set)
    test_df.sort_values(by="time", inplace=True)
    dev_df.sort_values(by="time", inplace=True)
    #保存
    train_df.to_csv("train.csv",index=0)
    test_df.to_csv("test.csv",index=0)
    dev_df.to_csv("dev.csv",index=0)