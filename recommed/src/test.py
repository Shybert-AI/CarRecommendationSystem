# -*- coding: UTF-8 -*-

import os
import sys
import pickle
import logging
import argparse
import numpy as np
import torch

from helpers import *
from models.general import *
from utils import utils
from models.sequential import *


def parse_global_args(parser):
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--random_seed', type=int, default=2021,
                        help='Random seed of numpy and pytorch.')
    parser.add_argument('--load', type=int, default=0,
                        help='Whether load model and continue to train')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--regenerate', type=int, default=0,
                        help='Whether to regenerate intermediate files.')
    return parser


def main():
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
               'regenerate', 'sep', 'train', 'verbose']
    logging.info(utils.format_arg_str(args, exclude_lst=exclude))  #格式化打印训练参数

    # Random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.info('cuda available: {}'.format(torch.cuda.is_available()))
    logging.info('# cuda devices: {}'.format(torch.cuda.device_count()))

    # Read data
    corpus_path = os.path.join(args.path, args.dataset, model_name.reader + '.pkl')
    corpus = pickle.load(open(corpus_path, 'rb'))
  
    # Define model
    model = model_name(args, corpus)
    logging.info(model)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()            
    # Run model
    data_dict = dict()
    history_items = [509, 515, 666, 691]
    data_dict["test"] = {'user_id': torch.tensor([6185]), 'item_id': torch.tensor([sorted(list(set(corpus.all_df["item_id"].values.tolist())))]), 'history_items': torch.tensor([history_items]), 'lengths': torch.tensor([4]), 'batch_size': 1, 'phase': 'test'}    
    
    runner = runner_name(args)
    prediction = model(data_dict["test"])
    reslut = torch.topk(prediction['prediction'],10)[1].tolist()[0]   #返回top10的商品
    reslut_list = []
    for i in reslut:   #移除用户购买过的商品
        if not i in history_items:
            reslut_list.append(i)
    logging.info(f"topk10的推荐结果为：{reslut_list}")        
    #print(reslut_list)  #返回推荐结果  [7889, 4381, 8417, 593, 3048, 7155, 7711, 611, 911, 2753]



if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--model_name', type=str, default='BPR', help='Choose a model to run.')
    init_args, init_extras = init_parser.parse_known_args()
    model_name = eval('{0}.{0}'.format(init_args.model_name)) #首先生成'SASRec.SASRec'，然后通过eval，变量要么定义要么导入，导入的话就是一个方法
    reader_name = eval('{0}.{0}'.format(model_name.reader))   #模型中定义了reader和runner类属性，因此可以调用，并且类属性在父类中
    runner_name = eval('{0}.{0}'.format(model_name.runner))

    # Args
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = reader_name.parse_data_args(parser)
    parser = runner_name.parse_runner_args(parser)
    parser = model_name.parse_model_args(parser)
    args, extras = parser.parse_known_args()

    # Logging configuration
    log_args = [init_args.model_name, args.dataset, str(args.random_seed)]
    for arg in ['epoch','lr', 'l2'] + model_name.extra_log_args:  #extra_log_args 类属性
        log_args.append(arg + '=' + str(eval('args.' + arg)))
    log_file_name = '__'.join(log_args).replace(' ', '__')
    if args.log_file == '':
        args.log_file = '../log/{}/{}.txt'.format(init_args.model_name, log_file_name)
    if args.model_path == '':
        args.model_path = '../model/{}/{}.pt'.format(init_args.model_name, log_file_name)

    utils.check_dir(args.log_file)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(init_args)

    main()
