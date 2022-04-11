import os
import pickle as pkl                 
import argparse
import os
import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import *
from config import *

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('-dataset', type=str, default='ogbl-biokg', help='dataset name, default to biokg')
    parser.add_argument('-model',  type=str, default='ComplEx')
    parser.add_argument('-mode',  type=str, default='search')
    parser.add_argument('-showTopConfig', action='store_true')
    parser.add_argument('-showAllConfig', action='store_true')
    parser.add_argument('-topNum', type=int, default=10)
    parser.add_argument('-query',  type=str, default='')
    parser.add_argument('-showValHistory',  action='store_true')
    parser.add_argument('-showEarlystoppingHistory',  action='store_true')
    parser.add_argument('-showOOMHistory',  action='store_true')
    parser.add_argument('-drawAllLearningCurves',  action='store_true')
    parser.add_argument('-showHPO',  action='store_true')
    parser.add_argument('-saveHPOtrials',  action='store_true')
    parser.add_argument('-showReduced',  action='store_true')
    parser.add_argument('-storeTopConfig',  type=str, default=None)
    parser.add_argument('-storeTopReduced',  type=str, default=None)
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    pklFile = './results/{}/{}_log.pkl'.format(args.dataset, args.mode)
    if not os.path.exists(pklFile):
        print('File not found: ', pklFile)
        exit()

    data = pkl.load(open(pklFile, 'rb'))
    mrr_list = []; test_mrr_list = []
    val_full_metrics = []; test_full_metrics = []
    configKey_list = set()
    config_list = []
    HPO_records = []

    for k,v in data.items():
        modelName = k.split('_')[0]
        if modelName != args.model:
            continue

        for res in v:
            try:
                mrr = float(res['evaluation']['mrr'])
            except:
                continue

            if mrr > 0:
                val_mrr_index = list(res['evaluation']['val_history'].values()).index(mrr)
                val_mrr_iters = list(res['evaluation']['val_history'].keys())[val_mrr_index]
                test_mrr      = res['evaluation']['test_history'][val_mrr_iters]

                configKey_list.add(k)
                config_list.append(res['config'])

                try:
                    val_full_metrics.append([res['evaluation'][key] for key in ['mrr', 'hits@1', 'hits@3', 'hits@10']])
                    mrr_list.append(mrr)
                except:
                    val_full_metrics.append([-1])
                    mrr_list.append(-1)

                try:
                    test_full_metrics.append([res['evaluation']['test_metrics'][key] for key in ['mrr', 'hits@1', 'hits@3', 'hits@10']])
                    test_mrr_list.append(test_mrr)
                except:
                    test_full_metrics.append([-1])
                    test_mrr_list.append(-1)

            if 'HPO_msg' in res['evaluation']:
                HPO_records.append(res)

    sorted_mrr = sorted(mrr_list)[::-1]
    print('\n==> dataset={}, model={}, finish {} config, {} experiments yet'.format(args.dataset, args.model, len(configKey_list), len(mrr_list)))
    if len(mrr_list) == 0:
        exit()

    # show top k configuration
    k = args.topNum
    topMrr = sorted_mrr[:k]
    print('==> max mrr for {} is : val={}, test={}'.format(args.dataset, max(topMrr), max(test_mrr_list)))
    best_test_index = test_mrr_list.index(max(test_mrr_list))
    print(f'==> full val metrics: {val_full_metrics[best_test_index]}\n==> full test metrics: {test_full_metrics[best_test_index]}')
    
    if args.showHPO:
        print('\n==> HPO results: ')
        HPO_dict = defaultdict(list)
        for record in HPO_records:
            HPO_dict[record['evaluation']['HPO_msg']].append(record)

        for HPO_msg, HPO_data in HPO_dict.items():
            data_num = len(HPO_data)
            val_mrr_list, test_mrr_list = [], [] 
            training_time_list, selected_cfg = [], []

            for res in HPO_data:
                try:
                    val_mrr       = float(res['evaluation']['mrr'])
                    test_mrr      = max(res['evaluation']['test_history'].values())
                except:
                    val_mrr, test_mrr = -1, -1

                if val_mrr > 0 and test_mrr > 0:
                    selected_cfg.append(res['config'])
                    val_mrr_list.append(val_mrr)
                    test_mrr_list.append(test_mrr)
                    training_time_list.append(res['evaluation']['trainting_time'])

            if len(val_mrr_list) > 0:
                print('*'*50, '\n', HPO_msg, f', finish {len(val_mrr_list)} yet')
                print(f'max val mrr = {max(val_mrr_list)} mean = {np.mean(val_mrr_list)}; max test mrr = {max(test_mrr_list)}, mean = {np.mean(test_mrr_list)}')
                print(f'val_mrr_list:  {val_mrr_list}')
                print(f'test_mrr_list: {test_mrr_list}')
                print(f'total training time: {sum(training_time_list)/3600} h')
                print(f'training time: {list(np.array(training_time_list)/3600)}')

                if args.saveHPOtrials:
                    folder = f'./results/{args.dataset}/trials/'
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    savePath = folder + f'{args.model}_{len(val_mrr_list)}_trials_{str(max(test_mrr_list))[:6]}.pkl'

                    data = {}
                    data['configs']            = selected_cfg
                    data['val_mrr']            = val_mrr_list
                    data['test_mrr']           = test_mrr_list
                    data['training_time_list'] = training_time_list
                    data['ref_dataset_names']  = [args.dataset for i in range(len(val_mrr_list))]

                    print(f'save trials to {savePath}')
                    pkl.dump(data, open(savePath, 'wb'))


    if args.showTopConfig:
        print('==> Top{} val mrr:\n'.format(k), topMrr)
        print('\n==> Top{} configuration:'.format(k))
        mrr_config_dict = {}

        for k,v in data.items():
            modelName = k.split('_')[0]
            if modelName != args.model:
                continue
            
            for res in v:
                try:
                    _ = res['evaluation']['mrr']
                except:
                    continue

                if res['evaluation']['mrr'] in topMrr:
                    val_history  = res['evaluation']['val_history']
                    test_history = res['evaluation']['test_history']
                    index        = list(val_history.values()).index(res['evaluation']['mrr'])
                    bestIters    = list(val_history.keys())[index]
                    test_mrr     = test_history[bestIters]
                    global_best_test_mrr = max(test_history.values())
                    mrr_config_dict[res['evaluation']['mrr']] = (res['config'], bestIters, test_mrr, global_best_test_mrr)

        for k in sorted(mrr_config_dict.keys())[::-1]:
            print('{} val mrr={}'.format(args.model, str(k)[:6]), mrr_config_dict[k])

        if args.storeTopConfig != None:
            print(f'==> storing top config to {args.storeTopConfig}')
            new_trials = []
            for k in sorted(mrr_config_dict.keys())[::-1]:
                cfg = mrr_config_dict[k][0]
                new_trials.append({'config':cfg, 'status':'none', 'val_mrr':-1, 'test_mrr':-1})

            pkl.dump(new_trials, open(args.storeTopConfig, 'wb'))
