import argparse
import json
import logging
import sys
import os
import time
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ogb.linkproppred import Evaluator
from tqdm import tqdm
from utils import *
from KGEModel import *
from config import *
from dataloader import *

def base_run_model(args, params_dict):
    # --------- checking and revising configure -------------------------------------------------
    # change param to log scale values if need
    for key in ['lr', 'regu_weight', 'embedding_range']:
        if key in params_dict.keys() and params_dict[key] < 0:
            params_dict[key] = 10 ** float(str(params_dict[key])[:10])
            
    if params_dict['n_neg'] in ['1VsAll', 'kVsAll']:
        params_dict['training_mode'] = params_dict['n_neg']
        params_dict['n_neg']         = 0
    else:
        params_dict['training_mode'] = 'negativeSampling'
        params_dict['n_neg']         = int(params_dict['n_neg'])

    if args.model == 'ConvE' and 'conve_drop1' not in params_dict.keys():  
        params_dict['conve_drop1'] = 0.0
        params_dict['conve_drop2'] = 0.0

    if args.model == 'TuckER' and 'tucker_drop1' not in params_dict.keys():  
        params_dict['tucker_drop1'] = 0.0
        params_dict['tucker_drop2'] = 0.0

    # check and synchronize configurations in params_dict and global_cfg
    def syncParamsAndGlobalCfg():
        keys = ['shareInverseRelation', 'initializer', 'loss_function', \
                    'dropoutRate', 'label_smooth', 'filter_falseNegative', 'optimizer', 'embedding_range', 
                    'n_neg', 'gamma', 'dim', 'batch_size', 'training_mode']
        if args.model == 'ConvE':  keys += ['conve_drop1', 'conve_drop2']
        if args.model == 'TuckER': keys += ['tucker_drop1', 'tucker_drop2']
        for key in keys:
            if key in params_dict.keys():
                # params_dict -> global_cfg
                exec('global_cfg.training_strategy.{} = params_dict[key]'.format(key))
            else:
                # global_cfg -> params_dict
                exec('params_dict[key] = global_cfg.training_strategy.{}'.format(key))

    params_dict = reviseConfig(params_dict) 
    syncParamsAndGlobalCfg()
    logging.info(f'==> params_dict: {params_dict} \n')

    # check training strategy
    if checkTrainingStrategy(args.model, args.dataset, global_cfg.training_strategy) != 'PASS':
        logging.info('==> [Error] fail in checkTrainingStrategy')
        return {'loss': 1.0, 'status': 'FAIL', 'val_mrr':-1, 'test_mrr':-1}

    # --------- finsih checking and revising configure -------------------------------------------

    # build config and search historical records
    searched_configs = getSearchedConfigs(args.perf_dict)
    onehotKey = generateKeyForConfig(args.model, params_dict) 
    savePath  = os.path.join('results', args.dataset, 'saveEmb',  onehotKey + '.pkl')
    evaluator = Evaluator(name = args.dataset) if (args.dataset in ['ogbl-biokg', 'ogbl-wikikg2']) else None

    # build model
    kge_model = KGEModel(
        model_name=args.model,
        dataset_name=args.dataset,
        nentity=args.datasetInfo['nentity'],
        nrelation=args.datasetInfo['nrelation'],
        params_dict=params_dict,
        config=global_cfg,
        evaluator=evaluator,
        args=args
        )

    if args.loadPretrain:
        pretainPath = args.pretrainPath if args.pretrainPath != None else savePath
        kge_model.loadEmbeddingFromFile(pretainPath)

    # calculate parameters
    num_parameters = sum(p.numel() for p in kge_model.parameters())
    logging.info('==> num_parameters for {}: {}'.format(args.model, num_parameters))

    # check config & load pretrained weights
    if onehotKey in searched_configs.keys():
        if not args.resume:
            # avoid searching for the same configuration
            logging.info(f'==> skip configuration: {onehotKey}')
            res = searched_configs[onehotKey][0]
            mrr = float(res['evaluation']['mrr'])

            if mrr > 0.0:
                mrr_index = list(res['evaluation']['val_history'].values()).index(mrr)
                mrr_iter  = list(res['evaluation']['val_history'].keys())[mrr_index]
                test_mrr  = res['evaluation']['test_history'][mrr_iter]
                return {'loss': (1-mrr), 'status': 'OK', 'val_mrr':mrr, 'test_mrr':test_mrr}
            else:
                return {'loss': (1-mrr), 'status': 'FAIL', 'val_mrr':-1, 'test_mrr':-1}
    
    # move to device
    kge_model = kge_model.cuda()

    # build dataloader
    train_dataloader = DataLoader(
        TrainDataset(args.datasetInfo,
                     trainMode = global_cfg.training_strategy.training_mode,
                     filter_falseNegative = global_cfg.training_strategy.filter_falseNegative,
                     negative_sample_size = int(params_dict['n_neg'])), 
                     batch_size=int(params_dict['batch_size']),
                     shuffle=True, 
                     num_workers=max(1, args.cpu_num),
                     collate_fn=TrainDataset.collate_fn)

    train_iterator = TrainDataset.one_shot_iterator(train_dataloader)

    # select optimizer
    current_learning_rate = params_dict['lr']
    if global_cfg.training_strategy.optimizer.lower() == 'adam':
        selected_optim = torch.optim.Adam
    elif global_cfg.training_strategy.optimizer.lower() == 'adagrad':
        selected_optim = torch.optim.Adagrad
    elif global_cfg.training_strategy.optimizer.lower() == 'sgd':
        selected_optim = torch.optim.SGD
    else:
        raise ValueError('Optimzer %s not supported' % global_cfg.training_strategy.optimizer)

    # build optimizer
    optimizer = selected_optim(filter(lambda p: p.requires_grad, kge_model.parameters()), lr=current_learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True) # ori
    
    # start training
    best_mrr       = 0
    test_mrr       = 0
    best_metrics   = {}
    FAIL_flag      = False # Out Of Memory / encounter NAN loss
    tolerate_times = 0
    tolerate_thres = 5 # early-stopping
    train_history, val_history, test_history = {}, {}, {}
    start_time = time.time()
    torch.cuda.empty_cache()
    
    for step in range(args.max_steps):
        try:
            # loss = kge_model.train_step(kge_model, optimizer, train_iterator)
            onestep_summary = kge_model.train_step(kge_model, optimizer, train_iterator)

            if 'NAN loss' in onestep_summary.keys():
                best_metrics = {'NAN loss':True, 'mrr':0.0}
                FAIL_flag     = True
                break

        except RuntimeError as exception:

            if "out of memory" in str(exception):
                logging.warning("out of memory in training step")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                best_metrics = {'OOM':True, 'mrr':0.0}
                
            FAIL_flag = True
            break
           
        # evaluate on validate/test set
        if (step+1) % args.valid_steps == 0:
            # evaluate on validate set
            try:
                metrics = kge_model.test_step(kge_model, args, 'validate')
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    logging.warning("out of memory in testing step")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    best_metrics = {'OOM':True, 'mrr':0.0}
                FAIL_flag = True
                break

            mrr = metrics['mrr']
            val_history[step+1] = mrr
            logging.info('==> No.{} iter, val mrr={}'.format(step+1, mrr))

            # automatically adjust learning rate w.r.t to the validate MRR
            scheduler.step(mrr)

            if mrr > best_mrr:
                best_mrr       = mrr
                best_metrics   = metrics
                tolerate_times = 0

                if args.saveEmbedding:
                    # kge_model.saveEmbeddingToFile(savePath)
                    savePathWithIter = os.path.join('results', args.dataset, 'saveEmb',  'iter_{}_valmrr_{}.pkl'.format(step+1, str(mrr)[:6]))
                    kge_model.saveEmbeddingToFile(savePathWithIter)
                    logging.info('iter={}, save embeddings to {}'.format(step, savePathWithIter))
                
                if args.eval_test:
                    try:
                        tst_metrics = kge_model.test_step(kge_model, args, 'test')
                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            logging.warning("out of memory in testing step")
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                            best_metrics = {'OOM':True, 'mrr':0.0}
                            FAIL_flag = True
                            break

                    best_metrics['test_metrics'] = tst_metrics
                    test_mrr                     = tst_metrics['mrr']
                    test_history[step+1]         = test_mrr
                    logging.info('==> No.{} iter, test mrr={}'.format(step+1, test_mrr))

            else:
                # early stopping 
                tolerate_times += 1
                if args.earlyStop and tolerate_times >= tolerate_thres:
                    best_metrics['early_stopping_step'] = step + 1
                    logging.info('==> early stopping at No.{} step'.format(step + 1))
                    break

    # add information to be saved locally
    best_metrics['val_history']    = val_history
    best_metrics['test_history']   = test_history
    best_metrics['train_history']  = train_history
    best_metrics['tolerate_times'] = tolerate_times
    best_metrics['trainting_time'] = float(time.time() - start_time)

    if args.search:
        # search mode, add HPO info 
        args.HPO_trials += 1
        best_metrics['HPO_msg']    = args.HPO_msg
        best_metrics['HPO_trials'] = args.HPO_trials
        saveToPklFile(onehotKey, params_dict, best_metrics, args.perf_dict)
    else:
        # evaluate mode
        saveToPklFile(onehotKey, params_dict, best_metrics, args.perf_dict)

    if FAIL_flag:
        return {'loss': (1-best_mrr), 'status': 'FAIL', 'val_mrr': -1, 'test_mrr': -1}
    else:
        return {'loss': (1-best_mrr), 'status': 'OK', 'val_mrr': best_mrr, 'test_mrr': test_mrr}
