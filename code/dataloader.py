#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import *

class TrainDataset(Dataset):
    def __init__(self, datasetInfo, trainMode, filter_falseNegative=False, negative_sample_size=64):
        self.datasetName          = datasetInfo['datasetName']
        self.len                  = datasetInfo['train_len']
        self.nentity              = datasetInfo['nentity']
        self.nrelation            = datasetInfo['nrelation']
        self.count                = datasetInfo['train_count']        # subsampling_weight
        self.entity_dict          = datasetInfo['entity_dict']        # for ogbl-biokg dataset
        self.triples              = datasetInfo['train_triples']      # list of (h,r,t)
        self.indexing_tail        = datasetInfo['indexing_tail']
        self.trainMode            = trainMode
        self.negative_sample_size = negative_sample_size
        self.filter_falseNegative = filter_falseNegative

        assert self.trainMode in ['negativeSampling', '1VsAll', 'kVsAll']
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        
        if self.datasetName == 'ogbl-wikikg2':
            ####### wikikg2 ########
            head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
            positive_sample      = [head, relation, tail]
            positive_sample      = torch.LongTensor(positive_sample)
            filter_mask          = torch.Tensor([-1]) 
            subsampling_weight   = self.count[(head, relation)] + self.count[(tail, (relation+self.nrelation)%(2*self.nrelation))]
            subsampling_weight   = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

            # generate negative samples
            negative_sample = torch.randint(0, self.nentity, (self.negative_sample_size,))
                
            # TODO: remove
            # generate labels (0/1) for 1(k) Vs All training mode
            if self.trainMode == 'kVsAll':
                tail_peers         = self.indexing_tail[idx]
                labels             = torch.zeros(self.nentity)
                labels[tail_peers] = 1
            else:
                labels = torch.Tensor([-1])

        elif self.datasetName == 'ogbl-biokg':
            ####### biokg #######
            head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
            head_type, tail_type = self.triples['head_type'][idx], self.triples['tail_type'][idx]
            positive_sample      = [head + self.entity_dict[head_type][0], relation, tail + self.entity_dict[tail_type][0]]
            positive_sample      = torch.LongTensor(positive_sample)
            filter_mask          = torch.Tensor([-1])   
            subsampling_weight   = self.count[(head, relation)] + self.count[(tail, (relation+self.nrelation)%(2*self.nrelation))]
            subsampling_weight   = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
            # subsampling_weight   = self.count[idx] 
            
            # generate negative samples
            if self.trainMode == 'negativeSampling':
                negative_sample = torch.randint(self.entity_dict[tail_type][0], self.entity_dict[tail_type][1], (self.negative_sample_size,))

                if self.filter_falseNegative:
                    # tails = self.indexing_tail[idx] 
                    tails = self.indexing_tail[(head, relation)]
                    filter_mask = torch.from_numpy(np.in1d(negative_sample, tails, invert=True)).int()

            else: 
                # 1VsAll or kVsAll, no needs for generating indexes
                negative_sample = torch.Tensor([-1])
        
            # TODO: remove
            # generate labels (0/1) for 1(k) Vs All training mode
            if self.trainMode == 'kVsAll':
                tail_peers         = self.indexing_tail[idx]
                labels             = torch.zeros(self.nentity)
                labels[tail_peers] = 1

            else:
                labels = torch.Tensor([-1])

        elif 'biokg' in self.datasetName and 'sampled' in self.datasetName:
            head, relation, tail, head_type, tail_type = self.triples[idx]
            subsampling_weight   = self.count[idx]    
            positive_sample      = torch.LongTensor((head, relation, tail))
            filter_mask          = torch.Tensor([-1])   
            labels               = torch.Tensor([-1])
            # generate negative samples
            if self.trainMode == 'negativeSampling':
                # non-redundant sampling
                negative_sample = torch.randint(self.entity_dict[tail_type][0], self.entity_dict[tail_type][1], (self.negative_sample_size,))
                if self.filter_falseNegative:
                    filter_mask = torch.from_numpy(
                        np.in1d(negative_sample, self.indexing_tail[idx], invert=True)
                        ).int()
            else: 
                # 1VsAll or kVsAll: not supported (OOM)
                exit()
            
        else:
            ###### datasets: wn18(rr) fb15k(237)  ######
            head, relation, tail = self.triples[idx]
            subsampling_weight   = self.count[idx]    
            positive_sample      = torch.LongTensor((head, relation, tail))
            filter_mask          = torch.Tensor([-1])        

            # generate negative samples
            if self.trainMode == 'negativeSampling':
                # non-redundant sampling
                negative_sample = torch.randperm(self.nentity)[:self.negative_sample_size]

                if self.filter_falseNegative:
                    filter_mask = torch.from_numpy(
                        np.in1d(negative_sample, self.indexing_tail[idx], invert=True)
                        ).int()

            else: 
                # 1VsAll or kVsAll, no needs for generating indexes
                negative_sample = torch.Tensor([-1])

            # generate labels (0/1) for 1(k) Vs All training mode
            if self.trainMode == 'kVsAll':
                tail_peers         = self.indexing_tail[idx]
                labels             = torch.zeros(self.nentity)
                labels[tail_peers] = 1

            else:
                labels = torch.Tensor([-1])

        return positive_sample, negative_sample, labels, filter_mask, subsampling_weight

    @staticmethod
    def collate_fn(data):
        positive_sample    = torch.stack([_[0] for _ in data], dim=0)
        negative_sample    = torch.stack([_[1] for _ in data], dim=0)
        labels             = torch.stack([_[2] for _ in data], dim=0)
        filter_mask        = torch.stack([_[3] for _ in data], dim=0)
        subsampling_weight = torch.cat([_[4]   for _ in data], dim=0)

        return positive_sample, negative_sample, labels, filter_mask, subsampling_weight
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
        
class TestDataset(Dataset):
    def __init__(self, split, args, random_sampling=False, entity_dict=None):
        self.datasetName     = args.datasetInfo['datasetName']
        self.neg_size        = args.neg_size_eval_train if random_sampling else -1
        self.random_sampling = random_sampling
        
        if split == 'validate':
            self.triples         = args.datasetInfo['valid_triples']
        elif split == 'test':
            self.triples         = args.datasetInfo['test_triples']
        else:
            self.triples         = args.datasetInfo['train_triples']
        
        if self.datasetName not in ['ogbl-biokg', 'ogbl-wikikg2']:
            if split == 'validate':
                self.filteredSamples = args.datasetInfo['valid_negSamples']
            elif split == 'test':
                self.filteredSamples = args.datasetInfo['test_negSamples']
            else:
                self.filteredSamples = args.datasetInfo['train_negSamples']

        self.len             = len(self.triples['head']) if (self.datasetName in ['ogbl-biokg', 'ogbl-wikikg2']) else len(self.triples)
        self.nentity         = args.datasetInfo['nentity']
        self.nrelation       = args.datasetInfo['nrelation']
        self.entity_dict     = args.datasetInfo['entity_dict']

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if self.datasetName == 'ogbl-wikikg2':
            head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
            positive_sample = torch.LongTensor((head, relation, tail))
            negative_sample = torch.cat([torch.LongTensor([tail]), torch.from_numpy(self.triples['tail_neg'][idx])])

        elif self.datasetName == 'ogbl-biokg':
            head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
            head_type, tail_type = self.triples['head_type'][idx], self.triples['tail_type'][idx]
            positive_sample = torch.LongTensor((head + self.entity_dict[head_type][0], relation, tail + self.entity_dict[tail_type][0]))

            negative_sample = torch.cat([torch.LongTensor([tail + self.entity_dict[tail_type][0]]), 
                            torch.from_numpy(self.triples['tail_neg'][idx] + self.entity_dict[tail_type][0])])

        elif 'biokg' in self.datasetName and 'sampled' in self.datasetName:
            head, relation, tail, head_type, tail_type = self.triples[idx]
            positive_sample = torch.LongTensor((head, relation, tail))
            negative_sample = torch.from_numpy(self.filteredSamples[idx])

            return positive_sample, negative_sample

        elif 'wikikg2' in self.datasetName and 'sampled' in self.datasetName:
            head, relation, tail = self.triples[idx]
            positive_sample = torch.LongTensor((head, relation, tail))
            negative_sample = torch.from_numpy(self.filteredSamples[idx])
            return positive_sample, negative_sample

        else:
            head, relation, tail = self.triples[idx]
            filter_bias          = self.filteredSamples[idx]
            positive_sample      = torch.LongTensor((head, relation, tail))     

            return positive_sample, filter_bias

        return positive_sample, negative_sample
    
    @staticmethod
    def getFilteredSamples(head, relation, tail, all_true_tail, nentity):
        '''
        (1,  tail_index) if invalid (negative triple)
        (-1, tail_index) if valid (exsiting triple)
        '''
        
        tails              = all_true_tail[(head, relation)]
        filter_bias        = np.ones(nentity)
        filter_bias[tails] *= (-1)
        filter_bias[tail]  = 1

        return torch.Tensor(filter_bias)

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        return positive_sample, negative_sample
    
    @staticmethod
    def collate_fn_with_bias(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        filter_bias     = torch.stack([_[1] for _ in data], dim=0)
        return positive_sample, filter_bias
