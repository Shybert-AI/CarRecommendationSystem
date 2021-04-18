# coding: utf-8

import os
import pandas as pd
import numpy as np



data_df = pd.read_csv("data.csv",sep=",")




n_users = data_df['user_id'].value_counts().size   #统计评论商品的用户(去重)
n_items = data_df['item_id'].value_counts().size
n_clicks = len(data_df)
min_time = data_df['time'].min()
max_time = data_df['time'].max()


np.random.seed(2021)
NEG_ITEMS = 99

out_df = data_df.drop_duplicates(['user_id', 'item_id', 'time'])        #去重
out_df.sort_values(by=['time', 'user_id', 'item_id'], inplace=True)    #按时间排序

uids = sorted(out_df['user_id'].unique())   #去重   
user2id = dict(zip(uids, range(1, len(uids) + 1))) #用户数值化
iids = sorted(out_df['item_id'].unique())
item2id = dict(zip(iids, range(1, len(iids) + 1))) #商品数值化

out_df['user_id'] = out_df['user_id'].apply(lambda x: user2id[x])
out_df['item_id'] = out_df['item_id'].apply(lambda x: item2id[x])
out_df = out_df.reset_index(drop=True)   #重置索引
print(out_df.head())


clicked_item_set = dict()
for user_id, seq_df in out_df.groupby('user_id'):   #分组聚合
    clicked_item_set[user_id] = set(seq_df['item_id'].values.tolist())
    
def generate_dev_test(data_df):
    result_dfs = []
    for idx in range(2):
        result_df = data_df.groupby('user_id').tail(1).copy()
        data_df = data_df.drop(result_df.index)   #result_df.index 获取数据的索引 drop删除索引的列 
        neg_items = np.random.randint(1, len(iids) + 1, (len(result_df), NEG_ITEMS))
        for i, uid in enumerate(result_df['user_id'].values):
            user_clicked = clicked_item_set[uid]
            for j in range(len(neg_items[i])):
                while neg_items[i][j] in user_clicked:  #生成负样本，负样本不在用户购买的商品中
                    neg_items[i][j] = np.random.randint(1, len(iids) + 1)
        result_df['neg_items'] = neg_items.tolist()
        result_dfs.append(result_df)
    return result_dfs, data_df


leave_df = out_df.groupby('user_id').head(1)
data_df = out_df.drop(leave_df.index)

[test_df, dev_df], data_df = generate_dev_test(data_df)
train_df = pd.concat([leave_df, data_df]).sort_index()

print(f"训练集为{len(train_df)}条,验证集为{len(dev_df)},测试集为{len(test_df)}")


train_df.to_csv('train.csv',index=0)
dev_df.to_csv('dev.csv', index=0)
test_df.to_csv('test.csv', index=0)






