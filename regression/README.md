1.������Դ  ���ֳ����׼۸�Ԥ��	
https://tianchi.aliyun.com/competition/entrance/231784/information
���ݺ���
SaleID	����ID��Ψһ����
name	�����������ƣ�������
regDate	����ע�����ڣ�����20160101��2016��01��01��
model	���ͱ��룬������
brand	����Ʒ�ƣ�������
bodyType	�������ͣ������γ���0��΢�ͳ���1�����ͳ���2����ͳ���3�����񳵣�4��˫��������5�����񳵣�6�����賵��7
fuelType	ȼ�����ͣ����ͣ�0�����ͣ�1��Һ��ʯ������2����Ȼ����3����϶�����4��������5���綯��6
gearbox	�����䣺�ֶ���0���Զ���1
power	���������ʣ���Χ [ 0, 600 ]
kilometer	��������ʻ�����λ��km
notRepairedDamage	��������δ�޸����𻵣��ǣ�0����1
regionCode	�������룬������
seller	���۷������壺0���Ǹ��壺1
offerType	�������ͣ��ṩ��0������1
creatDate	��������ʱ�䣬����ʼ����ʱ��
price	���ֳ����׼۸�Ԥ��Ŀ�꣩
vϵ������	��������������v0-14����15����������
2.���ݷ���   1.���ݷ���.ipynb
   1.ͨ������ȱʧֵ��['model', 'bodyType', 'fuelType', 'gearbox']�е�ȱʧֵΪ [1, 4506, 5981, 8680]�����ɾ��'bodyType', 'fuelType', 'gearbox'�� 
   2.ͨ���������ͷ�����"notRepairedDamage"���д���24324��"-"ֵ��������nan�����滻 
   3.�۲⵽offerType��seller�������������б��һ�㲻���Ԥ����ʲô�������������ɾ�� 
   4.Ԥ��۸��Ƿ�����̬�ֲ�����˽��ж����任
3.����Ԥ����
dataset.py
4.ʹ��xgboost���ֳ��۸�Ԥ��ģ�ͼ����� regress.py  
5.������ train_experiment.py
  1������Ԥ��۸��Ƿ�����̬�ֲ�����˽��ж����任���任����ǰ�������֤
  2���������ɷַ�����ģ���������ݽ�ģ
  �õ�Ԥ��۸���Ҫ���б任����ʹ�����ɷַ���
6.����ģ�� test.py