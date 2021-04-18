#数据集Browsing_History是自己随机生成的，考虑loss在下降，评价指标不下降，
#因此采用一个真实购买商品的记录信息RealRecord，当成浏览信息
#train
#基于用户商品的协同过滤
python main.py --model_name NCF --emb_size 64 --layers '[64]' --lr 5e-4 --l2 1e-7 --dropout 0.2 --dataset 'Browsing_History' --epoch 100
python main.py --model_name NCF --emb_size 64 --layers '[64]' --lr 5e-4 --l2 1e-7 --dropout 0.2 --dataset 'RealRecord' --epoch 100


#基于浏览记录的序列化推荐
#train
python main.py --model_name SASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'RealRecord' --epoch 100
#test
python test.py --model_name SASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'RealRecord' 