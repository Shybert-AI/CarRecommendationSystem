#���ݼ�Browsing_History���Լ�������ɵģ�����loss���½�������ָ�겻�½���
#��˲���һ����ʵ������Ʒ�ļ�¼��ϢRealRecord�����������Ϣ
#train
#�����û���Ʒ��Эͬ����
python main.py --model_name NCF --emb_size 64 --layers '[64]' --lr 5e-4 --l2 1e-7 --dropout 0.2 --dataset 'Browsing_History' --epoch 100
python main.py --model_name NCF --emb_size 64 --layers '[64]' --lr 5e-4 --l2 1e-7 --dropout 0.2 --dataset 'RealRecord' --epoch 100


#���������¼�����л��Ƽ�
#train
python main.py --model_name SASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'RealRecord' --epoch 100
#test
python test.py --model_name SASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'RealRecord' 