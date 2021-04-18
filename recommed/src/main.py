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
    if False:# not args.regenerate and os.path.exists(corpus_path):
        logging.info('Load corpus from {}'.format(corpus_path))
        corpus = pickle.load(open(corpus_path, 'rb'))
    else:
        corpus = reader_name(args)
        logging.info('Save corpus to {}'.format(corpus_path))
        pickle.dump(corpus, open(corpus_path, 'wb'))   #保存数据对象，它是一个类，pickle可以读取对象

    # Define model
    model = model_name(args, corpus)
    logging.info(model)
    model.apply(model.init_weights)  #apply做用在每一层
    model.actions_before_train()
    model.to(model.device)

    # Run model
    data_dict = dict()
    for phase in ['train', 'dev', 'test']:
        data_dict[phase] = model_name.Dataset(model, corpus, phase)
    runner = runner_name(args)
    logging.info('Test Before Training: ' + runner.print_res(model, data_dict['test']))
    if args.load > 0:
        model.load_model()
    if args.train > 0:
        runner.train(model, data_dict)
    logging.info(os.linesep + 'Test After Training: ' + runner.print_res(model, data_dict['test']))

    model.actions_after_train()
    logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)


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
