import time
import os
import random
import itertools
import logging
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.utils.data import DataLoader
from   collections import defaultdict
from   tqdm import tqdm
from   dataloader import TestDataset
from   utils import *

class KGEModel(nn.Module):
    def __init__(self, model_name, dataset_name, nentity, nrelation, params_dict, config, evaluator=None, args=None):
        super(KGEModel, self).__init__()
        '''
            KGEModel class
            components:
                - definition of KGE models 
                - train and test functions
        '''
        # checking parameters
        if model_name not in ['AutoSF', 'TransE', 'DistMult', 'ComplEx', 'RotatE', 'RESCAL', 'ConvE', 'TuckER']:
            raise ValueError('model %s not supported' % model_name)

        # build model
        self.model_name           = model_name
        self.dataset              = dataset_name.lower()
        self.config               = config
        self.nentity              = nentity
        self.nrelation            = nrelation if config.training_strategy.shareInverseRelation else 2*nrelation
        self.hidden_dim           = params_dict['dim']
        self.epsilon              = 2.0
        self.gamma                = nn.Parameter(torch.Tensor([params_dict['gamma']]), requires_grad=False)
        self.embedding_range      = nn.Parameter(torch.Tensor([params_dict['embedding_range']]), requires_grad=False)

        # set relation dimension according to specific model
        if model_name == 'RotatE':
            self.relation_dim = int(self.hidden_dim / 2)
        elif model_name == 'RESCAL':
            self.relation_dim = int(self.hidden_dim ** 2)
        else:
            self.relation_dim = self.hidden_dim

        self.entity_embedding     = nn.Parameter(torch.zeros(self.nentity, self.hidden_dim))
        self.relation_embedding   = nn.Parameter(torch.zeros(self.nrelation, self.relation_dim))
        self.evaluator            = evaluator
        
        # read essential training config (from global_config)
        self.dropoutRate          = config.training_strategy.dropoutRate
        self.dropout              = nn.Dropout(p=self.dropoutRate)
        self.training_mode        = config.training_strategy.training_mode
        self.shareInverseRelation = config.training_strategy.shareInverseRelation
        self.label_smooth         = config.training_strategy.label_smooth
        self.loss_name            = config.training_strategy.loss_function
        self.uni_weight           = config.training_strategy.uni_weight
        self.adv_sampling         = config.training_strategy.negative_adversarial_sampling
        self.filter_falseNegative = config.training_strategy.filter_falseNegative
        self.adv_temperature      = params_dict['advs']
        self.regularizer          = params_dict['regularizer']   # FRO NUC DURA None
        self.regu_weight          = params_dict['regu_weight']

        # setup candidate loss functions
        self.KLLoss               = nn.KLDivLoss(size_average=False)
        self.MRLoss               = nn.MarginRankingLoss(margin=float(self.gamma), reduction='none')
        self.CELoss               = nn.CrossEntropyLoss(reduction='none')
        self.BCELoss              = nn.BCEWithLogitsLoss(reduction='none')
        self.weightedBCELoss      = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.Tensor([self.nentity]))

        # initialize embedding
        self.init_embedding(config.training_strategy.initializer)
        self.model_func = {
            'AutoSF':   self.AutoSF,
            'TransE':   self.TransE,
            'DistMult': self.DistMult,
            'ComplEx':  self.ComplEx,
            'RotatE':   self.RotatE,
            'RESCAL':   self.RESCAL,
            'ConvE':    self.ConvE,
            'TuckER':   self.TuckER, 
        }

        if model_name == 'ConvE':
            # key: embedding dimension, value: hidden size of fc layer
            # notes that 9728 is special for hidden_dim=200
            fc_project_dict = {100:3648, 200:9728, 500:27968, 1000:58368, 2000:119168}
            self.hidden_drop      = nn.Dropout(config.training_strategy.conve_drop1)
            self.feature_map_drop = nn.Dropout2d(config.training_strategy.conve_drop2)
            self.emb_dim1         = 20
            self.emb_dim2         = self.hidden_dim // self.emb_dim1
            self.conv1            = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=False)
            self.bn0              = nn.BatchNorm2d(1)
            self.bn1              = nn.BatchNorm2d(32)
            self.bn2              = nn.BatchNorm1d(self.hidden_dim)
            self.fc               = nn.Linear(fc_project_dict[self.hidden_dim], self.hidden_dim) 
            self.bias             = nn.Parameter(torch.zeros(self.nentity))

        if model_name == 'TuckER':
            # d2, d1, d1
            self.W = torch.nn.Parameter(torch.tensor(
                np.random.uniform(-1, 1, (self.hidden_dim, self.hidden_dim, self.hidden_dim)), 
                dtype=torch.float, device="cuda", requires_grad=True))

            self.hidden_dropout1 = torch.nn.Dropout(config.training_strategy.tucker_drop1)
            self.hidden_dropout2 = torch.nn.Dropout(config.training_strategy.tucker_drop2)
            self.bn0 = torch.nn.BatchNorm1d(self.hidden_dim)
            self.bn1 = torch.nn.BatchNorm1d(self.hidden_dim)
        
    def init_embedding(self, init_method):
        if init_method == 'uniform':
            # Fills the input Tensor with values drawn from the uniform distribution
            nn.init.uniform_(
                tensor=self.entity_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item() )
            nn.init.uniform_(
                tensor=self.relation_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item() )
        
        elif init_method == 'xavier_norm':
            nn.init.xavier_normal_(tensor=self.entity_embedding)
            nn.init.xavier_normal_(tensor=self.relation_embedding)

        elif init_method == 'normal':
            # Fills the input Tensor with values drawn from the normal distribution
            nn.init.normal_(tensor=self.entity_embedding, mean=0.0, std=self.embedding_range.item())
            nn.init.normal_(tensor=self.relation_embedding, mean=0.0, std=self.embedding_range.item())

        elif init_method == 'xavier_uniform':
            nn.init.xavier_uniform_(tensor=self.entity_embedding)
            nn.init.xavier_uniform_(tensor=self.relation_embedding)

        return

    def forward(self, sample, mode='single'):
        '''
            3 available modes: 
                - single     : for calculating positive scores
                - neg_sample : for negative sampling
                - all        : for 1(k) vs All training 
        '''
        head_index, relation_index, tail_index = sample
        inv_relation_mask = torch.where(relation_index >= self.nrelation) if self.shareInverseRelation else None
        relation_index    = relation_index % self.nrelation if self.shareInverseRelation else relation_index
        head              = self.dropout(self.entity_embedding[head_index])
        relation          = self.dropout(self.relation_embedding[relation_index])
        tail              = self.dropout(self.entity_embedding if mode == 'all' else self.entity_embedding[tail_index])

        if self.model_name == 'ConvE':
            bias  = self.bias if mode == 'all' else self.bias[tail_index]
            score = self.ConvE(head, relation, tail, bias, mode=mode)
        else:
            score = self.model_func[self.model_name](head, relation, tail, inv_relation_mask=inv_relation_mask, mode=mode)
        
        return score

    def AutoSF(self, head, relation, tail, inv_relation_mask, mode='single'):
        # reference: https://github.com/AutoML-Research/AutoSF/tree/AutoSF-OGB
        hs = torch.chunk(head, 4, dim=-1)
        rs = torch.chunk(relation, 4, dim=-1)

        if  'biokg' in self.dataset:
            hr0 = hs[0] * rs[0]
            hr1 = hs[0] * rs[0] + hs[1] * rs[1]
            hr2 = hs[2] * rs[2]
            hr3 = hs[3] * rs[3]
        elif 'wikikg2' in self.dataset:
            hr0 = hs[0] * rs[0]
            hr1 = hs[1] * rs[1] - hs[3] * rs[1]
            hr2 = hs[2] * rs[0] + hs[3] * rs[3]
            hr3 = hs[1] * rs[2] + hs[2] * rs[2]
        else:
            exit()
        
        hrs = torch.cat([hr0,hr1,hr2,hr3], dim=-1)

        if mode == 'all':
            score = torch.mm(hrs, tail.transpose(0,1))
        else:
            if mode == 'single': self.regularizeOnPositiveSamples([head, relation, tail], hrs)
            score = torch.sum(hrs.unsqueeze(1) * tail, dim=-1)

        return score

    def TransE(self, head, relation, tail, inv_relation_mask, mode='single'):
        '''
            (h,r,t):     h + r = t
            (t,INV_r,h): t + (-r) = h, INV_r = -r
            ori: score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        '''
        if self.shareInverseRelation:
            relation[inv_relation_mask] = -relation[inv_relation_mask]

        if mode == 'all':
            score = (head + relation).unsqueeze(1) - tail.unsqueeze(0)
        else:
            if mode == 'single': self.regularizeOnPositiveSamples([head, relation, tail], (head + relation))
            score = (head + relation).unsqueeze(1) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)

        return score
    
    def DistMult(self, head, relation, tail, inv_relation_mask, mode='single'):
        if mode == 'all':
            # for 1(k) vs all: [B, dim] * [dim, N] -> [B, N]
            score = torch.mm(head * relation, tail.transpose(0,1))
        else:
            if mode == 'single': self.regularizeOnPositiveSamples([head, relation, tail], (head * relation))
            score = torch.sum((head * relation).unsqueeze(1) * tail, dim=-1)

        return score

    def regularizeOnPositiveSamples(self, embeddings, queries):
        '''
        available regularizer: 
            FRO / NUC / DURA / None
        inputs:
            embeddings: heads, relations, tails 
            queries:    combination of heads and relations
        '''

        self.regu = 0
        [heads, relations, tails] = embeddings

        if self.regularizer == 'FRO':
            # squared L2 norm
            self.regu += heads.norm(p = 2)**2     / heads.shape[0]
            self.regu += tails.norm(p = 2)**2     / tails.shape[0]
            self.regu += relations.norm(p = 2)**2 / relations.shape[0]
            
        elif self.regularizer == 'NUC':
            # nuclear 3-norm
            self.regu += heads.norm(p = 3)**3     / heads.shape[0]
            self.regu += tails.norm(p = 3)**3     / tails.shape[0]
            self.regu += relations.norm(p = 3)**3 / relations.shape[0]

        elif self.regularizer == 'DURA':
            # duality-induced regularizer for tensor decomposition models
            # regu = L2(φ(h,r)) + L2(t)
            self.regu += queries.norm(p = 2)**2 / queries.shape[0]
            self.regu += tails.norm(p = 2)**2   / tails.shape[0]

        else: 
            # None
            pass

        self.regu *= self.regu_weight
        return

    def ComplEx(self, head, relation, tail, inv_relation_mask, mode='single'):
        '''
        INV_r = Conj(r)
        '''
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_relation, im_relation = torch.chunk(relation, 2, dim=-1)

        if self.shareInverseRelation:
            im_relation = im_relation.clone()
            im_relation[inv_relation_mask] = - im_relation[inv_relation_mask]

        re_hrvec = re_head * re_relation - im_head * im_relation
        im_hrvec = re_head * im_relation + im_head * re_relation
        hr_vec   = torch.cat([re_hrvec, im_hrvec], dim=-1)

        # regularization on positive samples
        if mode == 'single': self.regularizeOnPositiveSamples([head, relation, tail], hr_vec)

        # <φ(h,r), t> -> score
        if mode == 'all':
            score = torch.mm(hr_vec, tail.transpose(0, 1))
        else:
            score = torch.sum(hr_vec.unsqueeze(1) * tail, dim=-1)

        return score

    def TuckER(self, head, relation, tail, inv_relation_mask, mode='single'):
        W_mat  = torch.mm(relation, self.W.view(relation.size(1), -1))
        W_mat  = W_mat.view(-1, head.size(1), head.size(1))
        if self.shareInverseRelation: W_mat[inv_relation_mask] = W_mat[inv_relation_mask].transpose(1,2)
        
        W_mat  = self.hidden_dropout1(W_mat)
        hr_vec = torch.bmm(head.unsqueeze(1), W_mat) 
        hr_vec = self.bn1(hr_vec.squeeze(1)).unsqueeze(1)
        hr_vec = self.hidden_dropout2(hr_vec)

        if mode == 'all':
            score = torch.mm(hr_vec.squeeze(1), tail.transpose(0,1))             # [B, N]
        else:
            if mode == 'single': self.regularizeOnPositiveSamples([head, relation, tail], hr_vec)
            score = torch.bmm(hr_vec, tail.transpose(1,2)).squeeze(1)            # [B, N]

        score = torch.sigmoid(score)
        return score

    def RotatE(self, head, relation, tail, inv_relation_mask, mode='single'):   
        '''
        INV_r = Conj(r)
        '''
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        # make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation / (self.embedding_range.item()/pi)
        re_relation    = torch.cos(phase_relation)
        im_relation    = torch.sin(phase_relation)

        if self.shareInverseRelation:
            im_relation[inv_relation_mask] = -im_relation[inv_relation_mask]

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        if mode == 'all':
            re_score = re_score.unsqueeze(1) - re_tail.unsqueeze(0)
            im_score = im_score.unsqueeze(1) - im_tail.unsqueeze(0)
        else:
            if mode == 'single':
                query = torch.cat([re_score, im_score], dim=-1)
                self.regularizeOnPositiveSamples([head, relation, tail], query)

            re_score = re_score.unsqueeze(1) - re_tail
            im_score = im_score.unsqueeze(1) - im_tail

        score = torch.stack([re_score, im_score], dim = 0)  # [2, B, N, relation_dim]
        score = score.norm(dim = 0)                         # [B, N, relation_dim]
        score = self.gamma.item() - score.sum(dim = 2)      # [B, N]

        return score

    def RESCAL(self, head, relation, tail, inv_relation_mask, mode='single'):
        '''
            f(h,r,t) = hT * Wr * t
        '''

        relation = relation.view(-1, self.hidden_dim, self.hidden_dim)           # [B, dim*dim] -> [B, dim, dim]
        
        if self.shareInverseRelation: relation[inv_relation_mask] = relation[inv_relation_mask].transpose(1,2)

        hr_vec   = torch.bmm(head.unsqueeze(1), relation)                        # [B, 1, dim]

        if mode == 'all':
            score = torch.mm(hr_vec.squeeze(1), tail.transpose(0,1))             # [B, N]
        else:
            # regularization on positive samples
            if mode == 'single': self.regularizeOnPositiveSamples([head, relation, tail], hr_vec)
            score = torch.bmm(hr_vec, tail.transpose(1,2)).squeeze(1)            # [B, N]
        
        return score

    def ConvE(self, head, relation, tail, bias, mode='single'):
        '''
            f(h,r,t) = < conv(h,r), t >
        '''
        head     = head.view(-1, 1, self.emb_dim1, self.emb_dim2)
        relation = relation.view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = torch.cat([head, relation], 2)
        # stacked_inputs = self.bn0(stacked_inputs)

        # φ(h,r) -> x
        x = self.conv1(stacked_inputs)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)        
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x) # [B, dim]

        if mode == 'all':
            score = torch.mm(x, tail.transpose(1,0))
        else:
            if mode == 'single':
                self.regularizeOnPositiveSamples([head, relation, tail], x)

            score = torch.bmm(x.unsqueeze(1), tail.transpose(1,2)).squeeze(1)
        
        return score

    def saveEmbeddingToFile(self, savePath):
        saveData = {}
        saveData['entity_embedding']   = self.entity_embedding.cpu()
        saveData['relation_embedding'] = self.relation_embedding.cpu()
        logging.info(f'save embedding tensor to: {savePath}')
        pkl.dump(saveData, open(savePath, "wb" ))
        return

    def loadEmbeddingFromFile(self, savePath):
        if not os.path.exists(savePath):
            logging.info(f'[Error] embedding file does not exist: {savePath}')
            return
        data = savePickleReader(savePath)
        self.entity_embedding   = nn.Parameter(data['entity_embedding'])
        self.relation_embedding = nn.Parameter(data['relation_embedding'])
        logging.info(f'successfully loaded pretrained embedding from: {savePath}')
        return

    def train_step(self, model, optimizer, train_iterator):
        # prepare
        model.train()
        optimizer.zero_grad()
        onestep_summary = {}

        # data preparing
        positive_sample, negative_sample, labels, filter_mask, subsampling_weight = next(train_iterator)

        if self.training_mode == '1VsAll':
            labels = torch.zeros(positive_sample.shape[0], self.nentity) 
            labels[list(range(positive_sample.shape[0])), positive_sample[:, 2]] = 1

        # move to device
        positive_sample    = positive_sample.cuda()
        negative_sample    = negative_sample.cuda()
        labels             = labels.cuda()
        filter_mask        = filter_mask.cuda() if self.filter_falseNegative else None
        subsampling_weight = subsampling_weight.cuda()

        # forward
        positive_score = model((positive_sample[:,0], positive_sample[:,1], positive_sample[:,2].unsqueeze(1)), mode='single')     # [B, 1]
        if self.training_mode == 'negativeSampling':    
            negative_score = model((positive_sample[:,0], positive_sample[:,1], negative_sample), mode='neg_sample')               # [B, N_neg]
        else:
            all_score = model((positive_sample[:,0], positive_sample[:,1], negative_sample), mode='all')                           # [B, N_neg]

        # Margin Ranking Loss (MR)
        if self.loss_name == 'MR':
            # only supporting training mode of negativeSampling
            target = torch.ones(positive_score.size()).cuda()
            loss   = self.MRLoss(positive_score, negative_score, target)
            loss   = (loss * filter_mask).mean(-1) if self.filter_falseNegative else loss.mean(-1)                                  # [B]
            loss   = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum())             # [1]

        # Binary Cross Entropy Loss (BCE) 
        elif self.loss_name == 'BCE_mean':
            if self.training_mode == 'negativeSampling':
                pos_label = torch.ones(positive_score.size()).cuda()
                neg_label = torch.zeros(negative_score.size()).cuda()
                
                # label smoothing
                pos_label = (1.0 - self.label_smooth)*pos_label + (1.0/self.nentity) if self.label_smooth > 0 else pos_label
                neg_label = (1.0 - self.label_smooth)*neg_label + (1.0/self.nentity) if self.label_smooth > 0 else neg_label
                pos_loss  = self.BCELoss(positive_score, pos_label).squeeze(-1)                                                     # [B]
                neg_loss  = self.BCELoss(negative_score, neg_label)                                                                 # [B, N_neg]
                neg_loss  = (neg_loss * filter_mask).mean(-1) if self.filter_falseNegative else neg_loss.mean(-1)                   # [B]
                loss      = pos_loss + neg_loss 
                loss      = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum()) 

            else:
                # label smoothing
                labels   = (1.0 - self.label_smooth)*labels + (1.0/self.nentity) if self.label_smooth > 0 else labels
                loss     = self.weightedBCELoss(all_score, labels).mean(dim=1)                                                     # [B]
                loss     = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum())

        elif self.loss_name == 'BCE_sum':
            if self.training_mode == 'negativeSampling':
                pos_label = torch.ones(positive_score.size()).cuda()
                neg_label = torch.zeros(negative_score.size()).cuda()
                
                # label smoothing
                pos_label = (1.0 - self.label_smooth)*pos_label + (1.0/self.nentity) if self.label_smooth > 0 else pos_label
                neg_label = (1.0 - self.label_smooth)*neg_label + (1.0/self.nentity) if self.label_smooth > 0 else neg_label
                pos_loss  = self.BCELoss(positive_score, pos_label).squeeze(-1)                                                     # [B]
                neg_loss  = self.BCELoss(negative_score, neg_label)                                                                 # [B, N_neg]
                neg_loss  = (neg_loss * filter_mask).sum(-1) if self.filter_falseNegative else neg_loss.sum(-1)                     # [B]
                loss      = pos_loss + neg_loss 
                loss      = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum())   
            else: # 1vsAll or kvsAll
                # label smoothing
                labels = (1.0 - self.label_smooth)*labels + (1.0/self.nentity) if self.label_smooth > 0 else labels
                loss   = self.BCELoss(all_score, labels).sum(dim=1)                                                              # [B]
                loss   = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum())

        elif self.loss_name == 'BCE_adv':
            # assert self.training_mode == 'negativeSampling'
            pos_loss = self.BCELoss(positive_score, torch.ones(positive_score.size()).cuda()).squeeze(-1)                          # [B]
            neg_loss = self.BCELoss(negative_score, torch.zeros(negative_score.size()).cuda())                                     # [B, N_neg]
            neg_loss = ( F.softmax(negative_score * self.adv_temperature, dim=1).detach() * neg_loss )

            if self.training_mode == 'negativeSampling' and self.filter_falseNegative:
                neg_loss  = (neg_loss * filter_mask).sum(-1) 
            else:
                neg_loss  =  neg_loss.sum(-1)                                                                                      # [B]

            loss     = pos_loss + neg_loss 
            loss     = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum())                                                                             
        
        # Cross Entropy (CE)
        elif self.loss_name == 'CE':
            if self.training_mode == 'negativeSampling':
                # note that filter false negative samples is not supported here
                cat_score = torch.cat([positive_score, negative_score], dim=1)
                labels    = torch.zeros((positive_score.size(0))).long().cuda()
                loss      = self.CELoss(cat_score, labels)
                loss      = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum())
            
            elif self.training_mode in ['1VsAll', 'kVsAll']:
                loss = self.KLLoss(F.log_softmax(all_score, dim=1), F.normalize(labels, p=1, dim=1))   
        
        if torch.isnan(loss):
            onestep_summary['NAN loss'] = True
            return onestep_summary

        if torch.is_tensor(self.regu) and not (torch.isinf(self.regu) or torch.isnan(self.regu)):
            loss += self.regu

        loss.backward()
        optimizer.step()

        return onestep_summary

    def test_step(self, model, args, split, random_sampling=False):

        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()
        select_collate_fn = TestDataset.collate_fn if ('ogb' in args.dataset) else TestDataset.collate_fn_with_bias

        # Prepare dataloader for evaluation
        test_dataset = DataLoader(
                TestDataset(split, args, random_sampling), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num), 
                collate_fn=select_collate_fn)

        if args.dataset in ['ogbl-biokg', 'ogbl-wikikg2']:
            test_logs = defaultdict(list)
            with torch.no_grad():
                for positive_sample, negative_sample in test_dataset:
                    
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    score = model((positive_sample[:,0], positive_sample[:,1], negative_sample), 'neg_sample')

                    # using ogb.evaluator for evaluation
                    batch_results = self.evaluator.eval({'y_pred_pos': score[:, 0], 'y_pred_neg': score[:, 1:]})
                    for metric in batch_results:
                        test_logs[metric].append(batch_results[metric])

                tmp_metrics = {}
                for metric in test_logs:
                    tmp_metrics[metric] = torch.cat(test_logs[metric]).mean().item()

                metrics = {}
                metrics['mrr']     = tmp_metrics['mrr_list']
                metrics['hits@1']  = tmp_metrics['hits@1_list']
                metrics['hits@3']  = tmp_metrics['hits@3_list']
                metrics['hits@10'] = tmp_metrics['hits@10_list']

        elif 'ogbl' in args.dataset:
            # ogbl sampling KG
            all_ranking = []
            with torch.no_grad():
                for positive_sample, negative_sample in test_dataset:
                    
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    score = model((positive_sample[:,0], positive_sample[:,1], negative_sample), 'neg_sample')
                    
                    # explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)
                    positive_arg = torch.zeros(score.shape[0])
                    tmp_ranking  = torch.nonzero(argsort.cpu() == positive_arg.unsqueeze(1))[:, 1].numpy() + 1

                    all_ranking += list(tmp_ranking)

            # calculate metrics
            all_ranking        = np.array(all_ranking)
            metrics            = {}
            metrics['mrr']     = np.mean(1/all_ranking)
            metrics['mr']      = np.mean(all_ranking)
            metrics['hits@1']  = np.mean(all_ranking<=1)
            metrics['hits@3']  = np.mean(all_ranking<=3)
            metrics['hits@10'] = np.mean(all_ranking<=10)

        else: 
            # other datasets
            all_ranking = []
            with torch.no_grad():
                for positive_sample, filter_bias in test_dataset:
                    positive_sample = positive_sample.cuda()
                    filter_bias     = filter_bias.cuda()

                    # forward
                    score = model((positive_sample[:,0], positive_sample[:,1], None), 'all')
                    score = score - torch.min(score, dim=1)[0].unsqueeze(1)
                    score *= filter_bias
                    
                    # explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)
                    positive_arg = positive_sample[:, 2] # indexes of target entities
                    
                    # obtain rankings for the batch
                    tmp_ranking = torch.nonzero(argsort == positive_arg.unsqueeze(1))[:, 1].cpu().numpy() + 1
                    all_ranking += list(tmp_ranking)

            # calculate metrics
            all_ranking        = np.array(all_ranking)
            metrics            = {}
            metrics['mrr']     = np.mean(1/all_ranking)
            metrics['mr']      = np.mean(all_ranking)
            metrics['hits@1']  = np.mean(all_ranking<=1)
            metrics['hits@3']  = np.mean(all_ranking<=3)
            metrics['hits@10'] = np.mean(all_ranking<=10)

        return metrics

