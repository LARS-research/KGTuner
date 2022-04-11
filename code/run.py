import argparse
import json
import logging
import sys
import os
import time
import numpy as np
import pickle as pkl
import torch
from ogb.linkproppred import LinkPropPredDataset, Evaluator
from tqdm import tqdm
from utils import *
from KGEModel import *
from config import *
from random_forest import *
from dataloader import *
from base_function import *

def run_model(param):
    return base_run_model(args=args, params_dict=param)

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    # data & device setting
    parser.add_argument('-dataset', type=str, default='wn18rr', help='dataset name')
    parser.add_argument('-data_path', type=str, default='./dataset/')
    parser.add_argument('-model', default='ComplEx', type=str)
    parser.add_argument('-cpu', '--cpu_num', default=2, type=int)
    parser.add_argument('-gpu', nargs='+', type=int, default=[])
    parser.add_argument('-earlyStop',  action='store_true')
    # training process setting
    parser.add_argument('-evaluate', action='store_true', help='evaluate mode')
    parser.add_argument('-search', action='store_true', help='search hyper-parameters')
    parser.add_argument('-resume', action='store_true', help='resume training or not')
    parser.add_argument('-saveEmbedding', action='store_true', help='save embedding to local files')
    parser.add_argument('-loadPretrain', action='store_true', help='load pretrain parameters or not')
    parser.add_argument('-pretrainPath', type=str, default=None)
    parser.add_argument('-eval_test',  action='store_true')
    parser.add_argument('-test_batch_size', default=16, type=int, help='valid/test batch size')
    parser.add_argument('-evaluate_times', default=1, type=int, help='repeat evaluation times')
    parser.add_argument('-max_steps', default=100000, type=int)
    parser.add_argument('-valid_steps', default=2500, type=int)
    parser.add_argument('-max_trials', default=200, type=int)
    parser.add_argument('-HPO_msg',  default='', type=str)
    parser.add_argument('-seed', default=1, type=int)
    parser.add_argument('-pretrain_dataset', default=None, type=str)
    parser.add_argument('-topNumToStore', default=10, type=int)  # for pretraining 
    parser.add_argument('-HPO_acq', default='BORE', type=str)  # ['max', 'EI', 'UCB', 'BORE']
    parser.add_argument('-space', default='full', type=str)
    
    return parser.parse_args(args)

def set_logger(log_file):
    '''
    save logs to checkpoint and console
    DEBUG INFO WARNING ERROR CRITICAL
    '''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)   
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    for h in logger.handlers:
        h.setFormatter(formatter)

    return logger

def main(args):
    # check 
    if args.search:
        run_mode = 'search' 
    elif args.evaluate:
        run_mode = 'evaluate'
    else:
        logging.error('==> [Error]: you need to select a mode in "search" or "evaluate"')
        exit()
    
    # check path
    save_paths = ['results', os.path.join('results', args.dataset), os.path.join('results', args.dataset, 'saveEmb')]
    for save_path in save_paths:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    args.perf_file = os.path.join('results', args.dataset,  'search_log.txt')
    args.perf_dict = os.path.join('results', args.dataset,  'search_log.pkl') # fix to this file
    logger = set_logger(args.perf_file)
    
    # load data
    logger.info('preparing training data ...')
    pklFile = os.path.join(args.data_path, args.dataset.replace('-', '_'), 'datasetInfo.pkl')
    if os.path.exists(pklFile):
        args.datasetInfo = savePickleReader(pklFile) 
    else:
        args.addInverseRelation = True
        args.datasetInfo = prepareData(args)
        pkl.dump(args.datasetInfo, open(pklFile, "wb"))
    logging.info('finish preparing')

    if args.evaluate:
        args.eval_test = True
        args.resume    = True
        exec('eval_params = global_cfg.{}_cfg.eval_params'.format(args.model))
        params = locals()['eval_params'][args.dataset]
        logging.info('Evaluation params for {}: {}'.format(args.model, params))

        test_mrr = []; val_mrr  = []
        for time in range(1, args.evaluate_times+1):
            
            torch.manual_seed(time)
            random.seed(time)
            np.random.seed(time)

            # train from scratch
            eval_result = run_model(params)
            assert eval_result['status'] == 'OK'

            # calculate metrics
            val_mrr.append( eval_result['val_mrr']) 
            test_mrr.append(eval_result['test_mrr'])
            logging.info(f'==> val_mrr  list: {str(val_mrr)}')
            logging.info(f'==> test_mrr list: {str(test_mrr)}')
            
            if time > 0:
                logging.info('==> eval times={}, test mrr: mean={}, std={}'.format(time, np.mean(test_mrr), np.std(test_mrr)))
                logging.info('==> eval times={}, val  mrr: mean={}, std={}'.format(time, np.mean(val_mrr),  np.std(val_mrr)))

    elif args.search:
        args.HPO_trials = 0
        sample_num      = 1e4
        meta_feature    = pkl.load(open('./dataset/graph_meta_features.pkl', 'rb'))

        if args.dataset not in meta_feature.keys(): 
            meta_feature[args.dataset] = np.array([0 for i in range(9)])

        if args.pretrain_dataset != None:
            topkConfigs, topkValMRR, topkTestMRR = get_all_configs(args.pretrain_dataset, args.model)
            ref_dataset_names = [args.pretrain_dataset for i in range(len(topkValMRR))]
            assert len(topkConfigs) > 0

            if args.pretrain_dataset not in meta_feature.keys(): 
                meta_feature[args.pretrain_dataset] = np.array([0 for i in range(9)])

            # convert config to correct format (lr, embedding_range, regu_weight) invert log
            for cfg in topkConfigs:
                if cfg['lr'] > 0:               cfg['lr']              = np.log10(cfg['lr'])
                if cfg['embedding_range'] > 0:  cfg['embedding_range'] = np.log10(cfg['embedding_range']) 
                if cfg['regu_weight'] > 0:      cfg['regu_weight']     = np.log10(cfg['regu_weight'])

            topkConfigs = [reviseConfigViaTrainingMode(cfg) for cfg in topkConfigs]

        assert args.space in ['full', 'reduced']
        assert args.HPO_acq in ['max', 'EI', 'UCB', 'BORE']
        if args.space == 'full':    
            selected_space = global_cfg.full_space

            # update 'dim' search range w.r.t. dataset and model
            if args.dataset == 'ogbl-wikikg2':
                selected_space['dim'] = ('choice', [100])
            else:
                if args.model == 'TuckER':
                    selected_space['dim'] = ('choice', [200, 500])
                elif args.model == 'RESCAL':
                    selected_space['dim'] = ('choice', [500, 1000])

        if args.space == 'reduced':  
            selected_space = global_cfg.reduced_space

        # searching 
        if args.pretrain_dataset != None:
            HPO_instance = RF_HPO(kgeModelName=args.model, obj_function=run_model, 
                                    dataset_name=args.dataset, HP_info=selected_space, acq=args.HPO_acq,
                                    meta_feature=meta_feature[args.dataset],
                                    msg=args.HPO_msg)

            # pretrain the surrogate model (RF)
            HPO_instance.pretrain_with_meta_feature(topkConfigs, topkValMRR, ref_dataset_names, meta_feature, topNumToStore=args.topNumToStore)
            # run HPO trials
            HPO_instance.runTrials(args.max_trials, sample_num, meta_feature=meta_feature[args.dataset])

        else:
            # run without pretrain data | pure exploration
            HPO_instance = RF_HPO(kgeModelName=args.model, obj_function=run_model, 
                                    dataset_name=args.dataset, HP_info=selected_space, acq=args.HPO_acq,
                                    meta_feature=None, msg=args.HPO_msg)
            # run HPO trials
            HPO_instance.runTrials(args.max_trials, sample_num)

if __name__ == '__main__':
    args = parse_args()
    args.gpu  = args.gpu[0] if args.gpu != [] else select_gpu()
    args.date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    args.HPO_msg += args.date

    logging.info('==> using No.{} GPU'.format(args.gpu))
    torch.cuda.set_device(args.gpu)
    torch.autograd.set_detect_anomaly(True)

    main(args)