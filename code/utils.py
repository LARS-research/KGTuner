import os
import sys
import inspect
import logging
import numpy as np
import torch
import copy
import pickle as pkl
import fcntl
import threading
import random
import subprocess
from random import randint
from tqdm import tqdm
from collections import defaultdict

def savePickleReader(file):
    if os.path.exists(file):
        while True:
            try:
                with open(file, "rb") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    unpickler = pkl.Unpickler(f)
                    data = unpickler.load()
                    f.close()
                    break
            except:
                continue
        return data
    else:
        return None

def savePickleWriter(data, file):
    with open(file, "wb") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        pkl.dump(data, f)
        f.close()
    return

def read_triple(file_path, entity2id, relation2id, nrelation, addInverseRelation=True):
    '''
    Read triples and map them into ids.
    Updates: augment dataset with inverse relation
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            ori_h, ori_r, ori_t = line.strip().split('\t')
            h, r, t = entity2id[ori_h], relation2id[ori_r], entity2id[ori_t]
            triples.append((h, r, t))

            if addInverseRelation:
                triples.append((t, r+nrelation, h))

    return triples

def addInverseRelation(triples, nrelation, withType=False):
    aug_triples = []
    if withType:
        for (h,r,t,h_type,t_type) in triples:
            aug_triples.append((t, r+nrelation, h, t_type, h_type))
    else:
        for (h,r,t) in triples:
            aug_triples.append((t, r+nrelation, h))

    return triples + aug_triples


def count_frequency(triples, nrelation, start=4, addInverseRelation=True):
    '''
    Get frequency of a partial triple like (head, relation) or (relation, tail)
    The frequency will be used for subsampling like word2vec
    '''
    count = {}
    # for head, relation, tail in triples:
    for d in triples:
        head, relation, tail = d[:3]
        if (head, relation) not in count:
            count[(head, relation)] = start
        else:
            count[(head, relation)] += 1

    if not addInverseRelation:
        head_count = {}
        for d in triples:
            head, relation, tail = d[:3]
            if (tail, relation) not in head_count:
                head_count[(tail, relation)] = start
            else:
                head_count[(tail, relation)] += 1

    countForTriple = []
    for d in triples:
        head, relation, tail = d[:3]
        if addInverseRelation:
            tmpCount = count[(head, relation)] + count[(tail, (relation+nrelation)%(2*nrelation))]
        else:
            tmpCount = count[(head, relation)] + head_count[(tail, relation)]
        countForTriple.append(torch.sqrt(1 / torch.Tensor([tmpCount])))

    return countForTriple

def get_true_tail(triples):
    '''
    Build a dictionary of true triples that will
    be used to filter these true triples for negative sampling
    '''

    true_tail = defaultdict(list)
    for d in triples:
        head, relation, tail = d[:3]
        true_tail[(head, relation)].append(tail)
        
    for head, relation in true_tail:
        true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))      

    return true_tail

def getFilteredSamples(triples, all_true_tail, nentity):
    '''
    (1,  tail_index) if invalid (negative triple)
    (-1, tail_index) if valid (exsiting triple)
    '''
    
    # filteredSamples = defaultdict(list)
    filteredSamples = []
    for (head, relation, tail) in tqdm(triples):
        tails              = all_true_tail[(head, relation)]
        filter_bias        = np.ones(nentity)
        filter_bias[tails] *= (-1)
        filter_bias[tail]  = 1
        negative_sample    = [ent for ent in range(nentity)]

        # filteredSamples.append([torch.LongTensor(negative_sample), torch.Tensor(filter_bias)])
        filteredSamples.append(torch.Tensor(filter_bias))

    return filteredSamples


def getIndexingTails(train_true_tail, train_triples):
    tailsByIndex = []
    for d in train_triples:
        head, relation, tail = d[:3]
        tailsByIndex.append(train_true_tail[(head, relation)])

    return tailsByIndex


def addInverseTriplesForOgblBiokg(triples, nrelation, dictConstant, withNegSamples=False):
    count, true_tail = defaultdict(dictConstant), defaultdict(list)
    inv_heads      = []
    inv_relations  = []
    inv_tails      = []
    inv_head_types = []
    inv_tail_types = []
    head_neg_list  = []
    tail_neg_list  = []

    # counting
    for i in tqdm(range(len(triples['head']))):
        head, relation, tail = triples['head'][i], triples['relation'][i], triples['tail'][i]
        head_type, tail_type = triples['head_type'][i], triples['tail_type'][i]

        if withNegSamples:
            head_neg, tail_neg = triples['head_neg'][i], triples['tail_neg'][i]
            head_neg_list.append(tail_neg)
            tail_neg_list.append(head_neg)

        # add inverse relation 
        inv_head, inv_relation, inv_tail = tail, relation + nrelation, head
        inv_head_type, inv_tail_type     = tail_type, head_type

        # counting
        count[(head, relation, head_type)] += 1
        count[(inv_head, inv_relation, inv_head_type)] += 1

        # get head/tail peers
        true_tail[(head, relation)].append(tail)
        true_tail[(inv_head, inv_relation)].append(inv_tail)

        inv_heads.append(inv_head)
        inv_relations.append(inv_relation)
        inv_tails.append(inv_tail)
        inv_head_types.append(inv_head_type)
        inv_tail_types.append(inv_tail_type)

    # append inverse triples to train set 
    triples['head']      = np.append(triples['head'],      inv_heads)
    triples['relation']  = np.append(triples['relation'],  inv_relations)
    triples['tail']      = np.append(triples['tail'],      inv_tails)
    triples['head_type'] = np.append(triples['head_type'], inv_head_types)
    triples['tail_type'] = np.append(triples['tail_type'], inv_tail_types)

    if withNegSamples:
        # print(triples['head_neg'].shape, np.array(head_neg_list).shape); exit()
        head_neg_list, tail_neg_list = np.array(head_neg_list), np.array(tail_neg_list)
        # print(triples['head_neg'].shape, triples['tail_neg'].shape)
        triples['head_neg'] = np.concatenate([triples['head_neg'], head_neg_list])
        triples['tail_neg'] = np.concatenate([triples['tail_neg'], tail_neg_list])
        # print(triples['head_neg'].shape, triples['tail_neg'].shape)

    # return triples, true_tail, count
    return triples


def addInverseTriplesForOgblWikikg2(triples, nrelation, dictConstant, withNegSamples=False):
    count, true_tail = defaultdict(dictConstant), defaultdict(list)
    inv_heads      = []
    inv_relations  = []
    inv_tails      = []
    head_neg_list  = []
    tail_neg_list  = []

    # counting
    for i in tqdm(range(len(triples['head']))):
        head, relation, tail = triples['head'][i], triples['relation'][i], triples['tail'][i]

        if withNegSamples:
            head_neg, tail_neg = triples['head_neg'][i], triples['tail_neg'][i]
            head_neg_list.append(tail_neg)
            tail_neg_list.append(head_neg)

        # add inverse relation 
        inv_head, inv_relation, inv_tail = tail, relation + nrelation, head

        # counting
        count[(head, relation)] += 1
        count[(inv_head, inv_relation)] += 1

        # get head/tail peers
        true_tail[(head, relation)].append(tail)
        true_tail[(inv_head, inv_relation)].append(inv_tail)

        inv_heads.append(inv_head)
        inv_relations.append(inv_relation)
        inv_tails.append(inv_tail)

    # append inverse triples to train set 
    triples['head']      = np.append(triples['head'],      inv_heads)
    triples['relation']  = np.append(triples['relation'],  inv_relations)
    triples['tail']      = np.append(triples['tail'],      inv_tails)

    if withNegSamples:
        head_neg_list, tail_neg_list = np.array(head_neg_list), np.array(tail_neg_list)
        triples['head_neg'] = np.concatenate([triples['head_neg'], head_neg_list])
        triples['tail_neg'] = np.concatenate([triples['tail_neg'], tail_neg_list])

    return triples

def addTailNegSamples(triples, all_true_tail, nentity, num=500, entity_dict=False, withType=False, filter=False):
    negSamples = []
    print('==> generating nagative samples')
    if withType:
        for (head, relation, tail, h_type, t_type) in tqdm(triples):
            true_tails = list(all_true_tail[(head, relation)])
            random_idx = list(range(entity_dict[t_type][0], entity_dict[t_type][1]))
            random.shuffle(random_idx)

            if filter:
                sample_neg = [tail]
                while len(sample_neg) < num:
                    tail = random_idx.pop()
                    if tail not in true_tails:
                        sample_neg.append(tail)
            else:
                sample_neg = [tail] + random_idx[:num]

            negSamples.append(sample_neg)
    else:
        for (head, relation, tail) in tqdm(triples):
            true_tails = list(all_true_tail[(head, relation)])
            random_idx = list(np.random.randint(0, nentity, num*10))
            random_idx = list(set(random_idx))

            if filter:
                sample_neg = [tail]
                while len(sample_neg) < num:
                    tail = random_idx.pop()
                    if tail not in true_tails:
                        sample_neg.append(tail)
            else:
                sample_neg = [tail] + random_idx[:num]

            negSamples.append(sample_neg)

    negSamples = np.array(negSamples)
    print('==> tail_neg.shape: {}'.format(negSamples.shape))

    return negSamples


def prepareData(args):
    datasetInfo, entity_dict, all_true_triples = dict(), dict(), dict()
    addInverseRelation_flag = args.addInverseRelation

    def dictConstant():
        return 4
    
    if args.dataset in ['ogbl-biokg', 'ogbl-wikikg2']:
        from ogb.linkproppred import LinkPropPredDataset
        dataset = LinkPropPredDataset(name = args.dataset)
        split_edge = dataset.get_edge_split()
        train_triples, valid_triples, test_triples = split_edge["train"], split_edge["valid"], split_edge["test"]

        if args.dataset == 'ogbl-biokg':
            cur_idx = 0
            for key in dataset[0]['num_nodes_dict']:
                entity_dict[key] = (cur_idx, cur_idx + dataset[0]['num_nodes_dict'][key])
                cur_idx += dataset[0]['num_nodes_dict'][key]

            nentity = sum(dataset[0]['num_nodes_dict'].values())
            nrelation = int(max(train_triples['relation'])) + 1
            valid_triples = addInverseTriplesForOgblBiokg(valid_triples, nrelation, dictConstant, withNegSamples=True)
            test_triples  = addInverseTriplesForOgblBiokg(test_triples,  nrelation, dictConstant, withNegSamples=True)
            train_triples = addInverseTriplesForOgblBiokg(train_triples, nrelation, dictConstant)

            train_true_tail = defaultdict(list)
            for i in tqdm(range(len(train_triples['head']))):
                head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
                head_type, tail_type = train_triples['head_type'][i], train_triples['tail_type'][i]
                train_true_tail[(head, relation)].append(tail)
            for head, relation in train_true_tail:
                train_true_tail[(head, relation)] = np.array(list(set(train_true_tail[(head, relation)])))   

            count = {}; 
            for i in tqdm(range(len(train_triples['head']))):
                head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
                head_type, tail_type = train_triples['head_type'][i], train_triples['tail_type'][i]

                if (head, relation) not in count:
                    count[(head, relation)] = 4
                else:
                    count[(head, relation)] += 1

            indexing_tail = train_true_tail
            train_count   = count

        elif args.dataset == 'ogbl-wikikg2':
            nentity       = dataset.graph['num_nodes']
            nrelation     = int(max(dataset.graph['edge_reltype'])[0]) + 1
            valid_triples = addInverseTriplesForOgblWikikg2(valid_triples, nrelation, dictConstant, withNegSamples=True)
            test_triples  = addInverseTriplesForOgblWikikg2(test_triples,  nrelation, dictConstant, withNegSamples=True)
            train_triples = addInverseTriplesForOgblWikikg2(train_triples, nrelation, dictConstant)
            
            count = {}; 
            for i in tqdm(range(len(train_triples['head']))):
                head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
                if (head, relation) not in count:
                    count[(head, relation)] = 4
                else:
                    count[(head, relation)] += 1
            
            indexing_tail = {}
            train_count   = count
    
    elif args.dataset in ['wn18rr', 'FB15k_237']:
        with open(os.path.join(args.data_path, args.dataset, 'entities.dict')) as fin:
            entity2id = dict()
            for line in fin:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)

        with open(os.path.join(args.data_path, args.dataset, 'relations.dict')) as fin:
            relation2id = dict()
            for line in fin:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)

        nentity           = len(entity2id)
        nrelation         = len(relation2id)
        
        # augment train data via inverse relation
        train_triples     = read_triple(os.path.join(args.data_path, args.dataset, 'train.txt'), entity2id, relation2id, nrelation, addInverseRelation_flag)
        valid_triples     = read_triple(os.path.join(args.data_path, args.dataset, 'valid.txt'), entity2id, relation2id, nrelation, addInverseRelation_flag)
        test_triples      = read_triple(os.path.join(args.data_path, args.dataset, 'test.txt'),  entity2id, relation2id, nrelation, addInverseRelation_flag)

        # All true triples
        all_true_triples  = set(train_triples + valid_triples + test_triples)

        # get head/tail peers
        train_true_tail   = get_true_tail(train_triples)
        all_true_tail     = get_true_tail(all_true_triples)
        indexing_tail     = getIndexingTails(train_true_tail, train_triples)
        
        # counting
        train_count       = count_frequency(train_triples, nrelation, addInverseRelation=addInverseRelation_flag)

        # get negative samples for evaluation
        valid_negSamples  = getFilteredSamples(valid_triples, all_true_tail, nentity)
        test_negSamples   = getFilteredSamples(test_triples,  all_true_tail, nentity)

    elif 'sampled' in args.dataset and 'ogb' in args.dataset:
        # loading synthetic data / other datasets
        savePath           = os.path.join(args.data_path, args.dataset, 'dataset.pkl')
        dataDict           = pkl.load(open(savePath, 'rb')) 
        nentity            = dataDict['nentity']      
        nrelation          = dataDict['nrelation']    
        train_triples      = dataDict['train_triples']
        valid_triples      = dataDict['valid_triples']
        test_triples       = dataDict['test_triples'] 
        entity_dict        = dataDict['entity_dict'] if 'biokg' in args.dataset else None
        nentity, nrelation = int(nentity), int(nrelation)

        # add inverse relation manully
        if addInverseRelation_flag:
            if 'biokg' in args.dataset:
                # biokg
                train_triples = addInverseRelation(train_triples, nrelation, withType=True)
                valid_triples = addInverseRelation(valid_triples, nrelation, withType=True)
                test_triples  = addInverseRelation(test_triples,  nrelation, withType=True)
            else:
                # wikikg2
                train_triples = addInverseRelation(train_triples, nrelation)
                valid_triples = addInverseRelation(valid_triples, nrelation)
                test_triples  = addInverseRelation(test_triples,  nrelation)

        # All true triples
        all_true_triples  = set(train_triples + valid_triples + test_triples)

        # get head/tail peers
        train_true_tail   = get_true_tail(train_triples)
        all_true_tail     = get_true_tail(all_true_triples)
        indexing_tail     = getIndexingTails(train_true_tail, train_triples)

        # add tail_neg for each val/test triple
        if 'biokg' in args.dataset:
            valid_negSamples = addTailNegSamples(valid_triples, all_true_tail, nentity, num=500, withType=True, entity_dict=entity_dict)
            test_negSamples  = addTailNegSamples(test_triples,  all_true_tail, nentity, num=500, withType=True, entity_dict=entity_dict)
        else:
            valid_negSamples = addTailNegSamples(valid_triples, all_true_tail, nentity, num=500)
            test_negSamples  = addTailNegSamples(test_triples,  all_true_tail, nentity, num=500)
        
        # counting
        train_count = count_frequency(train_triples, nrelation, addInverseRelation=addInverseRelation_flag)

    else: 
        # loading synthetic data / other datasets
        savePath      = os.path.join(args.data_path, args.dataset, 'dataset.pkl')
        dataDict      = pkl.load(open(savePath, 'rb'))
        nentity       = dataDict['nentity']      
        nrelation     = dataDict['nrelation']    
        train_triples = dataDict['train_triples']
        valid_triples = dataDict['valid_triples']
        test_triples  = dataDict['test_triples'] 
        nentity       = int(nentity)
        nrelation     = int(nrelation) 

        # add inverse relation manully
        if addInverseRelation_flag:
            train_triples = addInverseRelation(train_triples, nrelation)
            valid_triples = addInverseRelation(valid_triples, nrelation)
            test_triples  = addInverseRelation(test_triples,  nrelation)

        # All true triples
        all_true_triples  = set(train_triples + valid_triples + test_triples)

        # get head/tail peers
        train_true_tail   = get_true_tail(train_triples)
        all_true_tail     = get_true_tail(all_true_triples)
        indexing_tail     = getIndexingTails(train_true_tail, train_triples)
        
        # counting
        train_count       = count_frequency(train_triples, nrelation, addInverseRelation=addInverseRelation_flag)

        # get negative samples for val/test set
        if 'sampled_ogbl_wikikg2' in args.dataset:
            valid_negSamples  = [] # too costing
            test_negSamples   = []
        else:
            valid_negSamples  = getFilteredSamples(valid_triples, all_true_tail, nentity)
            test_negSamples   = getFilteredSamples(test_triples,  all_true_tail, nentity)

    datasetInfo['datasetName']      = args.dataset 
    datasetInfo['entity_dict']      = entity_dict
    datasetInfo['nentity']          = nentity
    datasetInfo['nrelation']        = nrelation
    datasetInfo['train_triples']    = train_triples
    datasetInfo['valid_triples']    = valid_triples
    datasetInfo['test_triples']     = test_triples
    datasetInfo['indexing_tail']    = indexing_tail
    datasetInfo['train_count']      = train_count
    datasetInfo['train_len']        = len(train_triples['head']) if args.dataset in ['ogbl-biokg', 'ogbl-wikikg2'] \
                                        else len(train_triples)

    if args.dataset not in ['ogbl-biokg', 'ogbl-wikikg2']:
        datasetInfo['valid_negSamples'] = valid_negSamples
        datasetInfo['test_negSamples']  = test_negSamples
        datasetInfo['all_true_tail']    = all_true_tail

    return datasetInfo

def saveTrialsToLocalFile(dataset, model, searchAlgorithm, trials):
    folder = f'./results/{dataset}/trials/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    savePath = f'./results/{dataset}/trials/{model}.pkl'
    if os.path.exists(savePath):
        data = savePickleReader(savePath)
    else:
        data = defaultdict(list)

    data[searchAlgorithm].append(trials)
    with open(savePath, 'wb') as f:
        pkl.dump(data, f)

    print(f'save trials to {savePath}')
    return 

def reviseConfig(tmp_config):
    if tmp_config['loss_function'] != 'BCE_adv':
        tmp_config['advs'] = 0

    if tmp_config['regularizer'] == 'None':
        tmp_config['regu_weight'] = 0

    return tmp_config

def checkTrainingStrategy(modelName, dataset, training_strategy):
    training_mode_flag = training_strategy.training_mode in ['1VsAll','kVsAll']
    
    # check datase
    if 'ogb' in dataset and training_mode_flag:
        return 'FAIL'
        
    # check model
    if modelName in ['TransE','RotatE','pRotatE'] and training_mode_flag:
        return 'FAIL'

    if training_strategy.loss_function in ['MR', 'BCE_adv'] and training_mode_flag:
        return 'FAIL'    

    # if not training_mode_flag and training_strategy.n_neg <= 0:
    #     return 'FAIL'  
    
    # if training_strategy.loss_function == 'MR' and training_strategy.gamma <= 0:
    #     return 'FAIL'   
    
    return 'PASS'

def generateKeyForConfig(modelName, params_dict):
    # using all config to build one-hot key
    key = str(modelName)
    for hp in sorted(params_dict.keys()):
        if hp == 'struct':
            key += '_struct'
            for s in params_dict['struct']:
                key += str(s)
        else:
            key += '_' + hp + str(params_dict[hp])

    return key

def getSearchedConfigs(pklFile):
    '''
        key: one-hot string representing a config (type: str)
        values: evaluation results on test set (type: dict)
    '''
    if os.path.exists(pklFile):
        # with open(pklFile, 'rb') as f:
        #     res = pkl.load(f)
        res = savePickleReader(pklFile)
    else:
        res = {}

    return res

def saveToPklFile(onehotKey, config, evalResults, pklFile):
    if os.path.exists(pklFile):
        # with open(pklFile, 'rb') as f:
        #     data = pkl.load(f)
        data = savePickleReader(pklFile)
    else:
        data = {}

    value = {}
    value['config']     = config
    value['evaluation'] = evalResults
    
    # records for more than one experiment 
    if onehotKey not in data.keys():
        data[onehotKey] = [value]
    else:
        data[onehotKey].append(value)
    
    pkl.dump(data, open(pklFile, "wb"))
    return

def generate_trial(tid, space):
    variables = space.keys()
    idxs = {v: [tid] for v in variables}
    vals = {k: [v] for k, v in space.items()}

    return {
        "state": 0,
        "tid": tid,
        "spec": None,
        "result": {"status": "new"},
        "misc": {
            "tid": tid,
            "cmd": ("domain_attachment", "FMinIter_Domain"),
            "workdir": None,
            "idxs": idxs,
            "vals": vals,
        },
        "exp_key": None,
        "owner": None,
        "version": 0,
        "book_time": None,
        "refresh_time": None,
    }

def checkConfigWithCondition(config, condition):
    pass_flag = True

    for k, v in condition.items():
        condition_type, value = v[0], v[1]
        
        if k == 'config_keys':
            if list(sorted(config.keys())) != value:
                # print(list(sorted(config.keys())), value)
                pass_flag = False
                break

        elif condition_type == 'equal' and config[k] != value:
            pass_flag = False
            break

        elif condition_type == 'in' and not config[k] in value:
            pass_flag = False
            break

    return pass_flag


def get_all_configs(dataset, model, mode='search', condition=None):
    pklFile = './results/{}/{}_log.pkl'.format(dataset, mode)
    if not os.path.exists(pklFile):
        print('File not found: ', pklFile)
        exit()

    logging.info(f'==> loading data for pre-training from: {pklFile}')
    data = pkl.load(open(pklFile, 'rb')) 
    config_list, val_mrr_list, test_mrr_list = [], [], []
    
    for k,v in data.items():
        modelName   = k.split('_')[0]
        config_dict = reviseConfigViaTrainingMode(v[0]['config'])
        
        if modelName != model:
            continue
        if condition != None and checkConfigWithCondition(config_dict, condition) == False:
            continue

        max_val_mrr, max_test_mrr = -1, -1
        for res in v:
            try:
                val_mrr = float(res['evaluation']['mrr'])
            except:
                continue

            if val_mrr > 0 and val_mrr > max_val_mrr:
                val_mrr_index = list(res['evaluation']['val_history'].values()).index(val_mrr)
                val_mrr_iters = list(res['evaluation']['val_history'].keys())[val_mrr_index]
                max_test_mrr  = res['evaluation']['test_history'][val_mrr_iters]
                max_val_mrr   = val_mrr

        if max_val_mrr > 0:
            val_mrr_list.append(max_val_mrr)
            test_mrr_list.append(max_test_mrr)
            config_list.append(config_dict)

    return config_list, val_mrr_list, test_mrr_list

def getTopkconfigs(dataset, model, mode='search', topK=10, condition=None):
    config_list, val_mrr_list, test_mrr_list = get_all_configs(dataset, model, mode, condition)
    topk_val_mrr  = sorted(val_mrr_list)[::-1][:topK]
    topk_index    = [val_mrr_list.index(i) for i in topk_val_mrr]
    topk_config   = [config_list[i] for i in topk_index]
    topk_test_mrr = [test_mrr_list[i] for i in topk_index]

    return topk_config, topk_val_mrr, topk_test_mrr

def getRandomConfigs(dataset, model, mode='search', num=10, condition=None):
    config_list, val_mrr_list, test_mrr_list = get_all_configs(dataset, model, mode, condition)
    random_index    = np.random.choice(range(len(config_list)), num, replace=False)
    random_val_mrr  = [val_mrr_list[i]  for i in random_index]
    random_config   = [config_list[i]   for i in random_index]
    random_test_mrr = [test_mrr_list[i] for i in random_index]

    return random_config, random_val_mrr, random_test_mrr

def getBatchTopConfigs(datasets, model, topK, condition):
    topkConfigs, topkValMRR, topkTestMRR, ref_dataset_names = [], [], [], []
    for dataset in datasets:
        cfg, valMRR, testMRR = getTopkconfigs(dataset, model, topK=topK, condition=condition)
        topkConfigs          += cfg
        topkValMRR           += valMRR
        topkTestMRR          += testMRR
        ref_dataset_names    += [dataset for i in range(len(cfg))]

    print(f'==> [getBatchTopConfigs] num. of topkConfigs={len(topkConfigs)}')

    return topkConfigs, topkValMRR, topkTestMRR, ref_dataset_names

def getBatchRandomConfigs(datasets, model, num, condition):
    randomConfigs, randomValMRR, randomTestMRR, ref_dataset_names = [], [], [], []
    for dataset in datasets:
        cfg, valMRR, testMRR = getRandomConfigs(dataset, model, num=num, condition=condition)
        randomConfigs       += cfg
        randomValMRR        += valMRR
        randomTestMRR       += testMRR
        ref_dataset_names   += [dataset for i in range(len(cfg))]

    print(f'==> [getBatchRandomConfigs] num. of randomConfigs={len(randomConfigs)}')

    return randomConfigs, randomValMRR, randomTestMRR, ref_dataset_names

def convertCategoryHPtoIndex(ori_config, HP_info):
    config = copy.deepcopy(ori_config)

    for HP_name, info in HP_info.items():
        HP_type, HP_range = info[0], info[1]
        if HP_type == 'choice':
            config[HP_name] = HP_range.index(config[HP_name])

            # if isinstance(HP_range[0], str):
            #     config[HP_name] = HP_range.index(config[HP_name])
            # elif isinstance(HP_range[0], bool):
            #     config[HP_name] = int(config[HP_name])
    
    return config

def reviseConfigViaTrainingMode(config):
    '''
    '32', '128', '512', '2048', 'kVsAll','1VsAll'
    '''
    if 'training_mode' not in config.keys():
        config['n_neg'] = str(config['n_neg'])
        return config

    training_mode = config.pop('training_mode')
    if 'All' in training_mode:
        config['n_neg'] = training_mode
    else:
        config['n_neg'] = str(config['n_neg'])

    return config

def configMatching(cfg1, cfg2):
    v1 = cfg1.keys()
    v2 = cfg2.keys()

    if len(v1) != len(v2):
        return False

    for key in v1:
        if isinstance(cfg1[key], float):
            if abs(cfg1[key] - cfg2[key]) > 1e-8: 
                print(key, cfg1[key], cfg2[key])
                return False
        else:
            if (cfg1[key] != cfg2[key]):
                print(key, cfg1[key], cfg2[key])
                return False

    return True

def newSummaryDict():
    newDict = {
            'pos_score':[], 
            'neg_score':[], 
            'all_score':[], 
            'pos_loss':[], 
            'neg_loss':[], 
            'total_loss':[], 
            }

    return newDict

def convert_to_ax_parameters(HP_info):
    ax_parameters = []
    for HP_name, info in HP_info.items():
        HP_type, HP_range = info[0], info[1]
        if HP_type == 'choice':
            if len(HP_range) == 1:
                continue
            tmp_info = {'name':HP_name, 'type':'choice', 'values':list(HP_range)}
        else:
            tmp_info = {'name':HP_name, 'type':'range', 'bounds':[HP_range[0], HP_range[1]]}

        ax_parameters.append(tmp_info)

    return ax_parameters

def fillWithFixedValues(config_list, HP_info):
    for cfg in config_list:
        for HP_name, info in HP_info.items():
            if HP_name in cfg.keys():
                continue
            cfg[HP_name] = info[1][0]

    return config_list
    
def selectRecordsWithTimeBudgets(data, budget=12):
    time_list  = data['training_time_list']
    accumulated_time_list = [sum(time_list[:i+1])/3600 for i in range(len(time_list))]
    # print(time_list, '\n', accumulated_time_list)
    accumulated_time_list = np.array(accumulated_time_list)

    try:
        final_index = np.where(accumulated_time_list > budget)[0][0] - 1
    except:
        final_index = len(accumulated_time_list)
    # print(final_index, accumulated_time_list[final_index], accumulated_time_list[final_index+1])

    topkConfigs       = data['configs'][:final_index]
    topkValMRR        = data['val_mrr'][:final_index]
    topkTestMRR       = data['test_mrr'][:final_index]
    ref_dataset_names = data['ref_dataset_names'][:final_index]

    return topkConfigs, topkValMRR, topkTestMRR, ref_dataset_names

def getTopConfigWithBudgets(pklFile, model, condition=None, topNum=-1, maxIter=2e4):
    # pklFile = '{}/{}/search_log.pkl'.format(folder, dataset)
    data = pkl.load(open(pklFile, 'rb'))
    mrr_list = []; test_mrr_list = []
    config_list = []
    
    for k,v in data.items():
        modelName = k.split('_')[0]
        if modelName != model:
            continue

        for res in v:
            
            PASS = True
            if condition != None:
                for k, v in condition.items():
                    if k in res['config'].keys() and res['config'][k] != v: 
                        PASS = False
                        break
            if not PASS: continue
            try:
                mrr = float(res['evaluation']['mrr'])
            except:
                continue

            if mrr > 0:
                valid_iters   = [iters for iters in res['evaluation']['val_history'].keys() if iters <= maxIter]
                valid_mrr     = [res['evaluation']['val_history'][iters] for iters in valid_iters]
                val_mrr_index = valid_mrr.index(max(valid_mrr))
                val_mrr_iters = valid_iters[val_mrr_index]
                test_mrr      = res['evaluation']['test_history'][val_mrr_iters]
                
                mrr_list.append(max(valid_mrr))
                test_mrr_list.append(test_mrr)
                config_list.append(res['config'])
                                   
    print(f'valid exp num: {len(mrr_list)}')

    if topNum != -1:
        mrr_list       = np.array(mrr_list)
        test_mrr_list  = np.array(test_mrr_list)
        config_list    = np.array(config_list)
        index          = np.argsort(test_mrr_list)[::-1][:topNum]
        return config_list[index], mrr_list[index], test_mrr_list[index]

    else:
        return config_list, mrr_list, test_mrr_list

def select_gpu():
    nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
    gpu_info = False
    gpu_info_line = 0
    proc_info = False
    gpu_mem = []
    gpu_occupied = set()
    i = 0
    for line in nvidia_info.stdout.split(b'\n'):
        line = line.decode().strip()

        if gpu_info:
            gpu_info_line += 1
            if line == '':
                gpu_info = False
                continue
            # if gpu_info_line % 3 == 2:
            #     if not ('RTX' in line or 'GTX' in line or ''):
            #         try: 
            #             mem_info = line.split('|')[2]
            #             used_mem_mb = int(mem_info.strip().split()[0][:-3])
            #             gpu_mem.append(used_mem_mb)
            #         except:
            #             continue

            if not ('RTX' in line or 'GTX' in line or ''):
                try: 
                    mem_info = line.split('|')[2]
                    used_mem_mb = int(mem_info.strip().split()[0][:-3])
                    gpu_mem.append(used_mem_mb)
                except:
                    continue
        
        if proc_info:
            if line == '|  No running processes found                                                 |':
                continue
            if line == '+-----------------------------------------------------------------------------+':
                proc_info = False
                continue
            proc_gpu = int(line.split()[1])
            #proc_type = line.split()[3]
            gpu_occupied.add(proc_gpu)
        
        if line == '|===============================+======================+======================|':
            gpu_info = True
        
        if line == '|=============================================================================|':
            proc_info = True
        i += 1

    for i in range(0,len(gpu_mem)):
        if i not in gpu_occupied:
            print('Automatically selected GPU Np.{} because it is vacant.'.format(i))
            occupy = torch.zeros((8,)).cuda(i)
            return i

    for i in range(0,len(gpu_mem)):
        # print(gpu_mem)
        if gpu_mem[i] == min(gpu_mem):
            print('All GPUs are occupied. Automatically selected GPU No.{} because it has the most free memory.'.format(i))
            occupy = torch.zeros((8,)).cuda(i)
            return i

if __name__ == '__main__':
    print(select_gpu())