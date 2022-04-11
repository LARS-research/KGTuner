import os
import pickle as pkl                 
import argparse
import os
import sys
import inspect
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from scipy.sparse import csr_matrix
from utils import *

'''
This script is use to generate sampled KG via random walk
'''

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='wn18rr', help='dataset name')
    parser.add_argument('-data_path', type=str, default='./dataset/')
    parser.add_argument('-sample_ratio', type=float, default=0.2)
    parser.add_argument('-repeat', type=int, default=0)
    parser.add_argument('-num_starts', type=int, default=10)
    return parser.parse_args(args)

class KGGenerator_randomWalks:
    def __init__(self, all_triples, sample_ratio, num_starts, savePath, entity_dict=None, repeat_num=0, split_ratio=[0.9, 0.05, 0.05]):        
        self.split_ratio  = split_ratio
        self.sample_ratio = sample_ratio
        self.repeat_num   = repeat_num
        self.num_starts   = num_starts

        # setup graph
        print('==> building nx graph...')
        homoGraph = self.triplesToNxGraph(all_triples)
        diGraph   = self.triplesToNxDiGraph(all_triples)
        print('==> Done!')

        # sampling via random walk
        print('==> start random walk sampling...')
        num_nodes = homoGraph.number_of_nodes()
        target_num_nodes = int(num_nodes * self.sample_ratio)
        if num_starts == 1:
            sampled_nodes = self.random_walk_induced_graph_sampling(homoGraph, target_num_nodes)
        else:
            sampled_nodes = self.multi_starts_random_walk_induced_graph_sampling(homoGraph, target_num_nodes, num_starts)
        sampled_graph = diGraph.subgraph(sampled_nodes)
        print('==> Done!')

        # build sampled KG
        self.all_triples = []
        self.relations   = []
        self.entities    = []
        for edge in list(sampled_graph.edges(data=True)):
            h,t = edge[0], edge[1]
            r = edge[2]['relation']
            self.all_triples.append((h,r,t))
            self.relations.append(r)
            self.entities.append(h)
            self.entities.append(t)

        # assign new index to entities/relation
        self.entities  = sorted(list(set(self.entities)))
        self.nentity   = len(self.entities)
        self.relations = sorted(list(set(self.relations)))
        self.nrelation = len(self.relations)
        self.ntriples  = len(self.all_triples)
        self.sparsity  = self.ntriples / (self.nentity * self.nentity * self.nrelation)
        self.entity_mapping_dict = {}
        self.relation_mapping_dict = {}

        print('dataset={}, nentity={}, sampled ratio={}, sparsity={}'.format(
            args.dataset, self.nentity, self.sample_ratio, self.sparsity))

        if 'biokg' in args.dataset:
            # using entity dict for generation
            entity_counts, new_entity_dict = {}, {}
            for key in ['disease', 'drug', 'function', 'protein', 'sideeffect']:
                entity_counts[key] = 0

            for idx in range(self.nentity):
                self.entity_mapping_dict[self.entities[idx]] = idx            
                entity_type = entity_dict[self.entities[idx]]
                entity_counts[entity_type] += 1
                new_entity_dict[idx] = entity_type

            self.entity_dict = {}
            cur_idx = 0
            for key in ['disease', 'drug', 'function', 'protein', 'sideeffect']:
                self.entity_dict[key] = (cur_idx, cur_idx + entity_counts[key])
                cur_idx += entity_counts[key]

            for idx in range(self.nrelation):
                self.relation_mapping_dict[self.relations[idx]] = idx

            # get new triples via entitie_mapping_dict
            self.all_new_triples = []
            for (h,r,t) in self.all_triples:
                new_h, new_t = self.entity_mapping_dict[h], self.entity_mapping_dict[t]
                new_r = self.relation_mapping_dict[r]
                h_type, t_type = new_entity_dict[new_h], new_entity_dict[new_t]
                new_triples = (new_h, new_r, new_t, h_type, t_type)
                self.all_new_triples.append(new_triples)

        else:
            # key:   origin index
            # value: new assigned index
            for idx in range(self.nentity):
                self.entity_mapping_dict[self.entities[idx]] = idx
            for idx in range(self.nrelation):
                self.relation_mapping_dict[self.relations[idx]] = idx

            # get new triples via entitie_mapping_dict
            self.all_new_triples = []
            for (h,r,t) in self.all_triples:
                new_h, new_t = self.entity_mapping_dict[h], self.entity_mapping_dict[t]
                new_r = self.relation_mapping_dict[r]
                new_triples = (new_h, new_r, new_t)
                self.all_new_triples.append(new_triples)

        # shuffle triples
        random.shuffle(self.all_new_triples)

        # split and save data
        self.trainset, self.valset, self.testset = self.splitData()
        
        # save dataset to local file
        self.saveData(savePath)

    @staticmethod
    def triplesToNxGraph(triples):
        # note that triples are with no inverse relations
        graph = nx.Graph()
        nodes = list(set([h for (h,r,t) in triples] + [t for (h,r,t) in triples]))
        graph.add_nodes_from(nodes)
        edges = list(set([(h,t) for (h,r,t) in triples]))
        graph.add_edges_from(edges)

        return graph

    @staticmethod
    def triplesToNxDiGraph(triples):
        # note that triples are with no inverse relations
        graph = nx.MultiDiGraph()
        nodes = list(set([h for (h,r,t) in triples] + [t for (h,r,t) in triples]))
        graph.add_nodes_from(nodes)

        for (h,r,t) in triples:
            graph.add_edges_from([(h,t)], relation=r)

        return graph
        
    @staticmethod
    def random_walk_induced_graph_sampling(complete_graph, nodes_to_sample):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n

        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = nodes_to_sample
        index_of_first_random_node = random.randint(0, nr_nodes - 1)
        Sampled_nodes = set([complete_graph.nodes[index_of_first_random_node]['id']])

        iteration   = 1
        growth_size = 2
        check_iters = 100
        nodes_before_t_iter = 0
        curr_node = index_of_first_random_node; print(f'==> curr_node: {curr_node}')
        while len(Sampled_nodes) != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            index_of_edge = random.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            Sampled_nodes.add(complete_graph.nodes[chosen_node]['id'])
            curr_node = chosen_node
            iteration = iteration + 1

            if iteration % check_iters == 0:
                if ((len(Sampled_nodes) - nodes_before_t_iter) < growth_size):
                    print(f'==> boost seaching, skip to No.{curr_node} node')
                    curr_node = random.randint(0, nr_nodes - 1)
                nodes_before_t_iter = len(Sampled_nodes)

        return Sampled_nodes

    @staticmethod
    def multi_starts_random_walk_induced_graph_sampling(complete_graph, nodes_to_sample, num_starts):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)

        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n

        nr_nodes        = len(complete_graph.nodes())
        start_candidate = [random.randint(0, nr_nodes - 1) for i in range(num_starts)]
        Sampled_nodes   = set()

        for idx, index_of_first_random_node in enumerate(start_candidate):
            Sampled_nodes.add(complete_graph.nodes[index_of_first_random_node]['id'])
            iteration           = 1
            growth_size         = 2
            check_iters         = 100
            nodes_before_t_iter = 0
            target_num = int((idx+1) * nodes_to_sample / num_starts)
            curr_node  = index_of_first_random_node

            while len(Sampled_nodes) < target_num:
                edges = [n for n in complete_graph.neighbors(curr_node)]
                index_of_edge = random.randint(0, len(edges) - 1)
                chosen_node = edges[index_of_edge]
                Sampled_nodes.add(complete_graph.nodes[chosen_node]['id'])
                curr_node = chosen_node
                iteration = iteration + 1

                if iteration % check_iters == 0:
                    if ((len(Sampled_nodes) - nodes_before_t_iter) < growth_size):
                        print(f'==> boost seaching, skip to No.{curr_node} node')
                        curr_node = random.randint(0, nr_nodes - 1)
                    nodes_before_t_iter = len(Sampled_nodes)

        return Sampled_nodes

    def splitData(self):
        '''
            split triples with certain ratio stored in self.split_ratio
        '''
        n1 = int(self.ntriples * self.split_ratio[0])
        n2 = int(n1 + self.ntriples * self.split_ratio[1])
        return self.all_new_triples[:n1], self.all_new_triples[n1:n2], self.all_new_triples[n2:]

    def saveData(self, savePath):
        if self.repeat_num == 0:
            if self.num_starts == 1:
                folder = 'sampled_{}_{}'.format(args.dataset, self.sample_ratio)
            else:
                folder = 'sampled_{}_{}_starts_{}'.format(args.dataset, self.sample_ratio, self.num_starts)
        else:
            if self.num_starts == 1:
                folder = 'sampled_{}_{}_rp{}'.format(args.dataset, self.sample_ratio, self.repeat_num)
            else:
                folder = 'sampled_{}_{}_starts_{}_rp{}'.format(args.dataset, self.sample_ratio, self.num_starts, self.repeat_num)

        saveFolder = os.path.join(savePath, folder)
        saveFolder = saveFolder.replace('-', '_')
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)

        dataDict = {}
        dataDict['nentity']       = self.nentity
        dataDict['nrelation']     = self.nrelation
        dataDict['train_triples'] = self.trainset
        dataDict['valid_triples'] = self.valset
        dataDict['test_triples']  = self.testset
        dataDict['entity_mapping_dict'] = self.entity_mapping_dict
        dataDict['relation_mapping_dict'] = self.relation_mapping_dict
        if 'biokg' in args.dataset: dataDict['entity_dict'] = self.entity_dict

        dictPath = os.path.join(saveFolder, 'dataset.pkl')
        print('==> save to:', dictPath)
        pkl.dump(dataDict, open(dictPath, "wb" ))


if __name__ == '__main__':
    args = parse_args()

    print('==> loading dataset ...')
    pklFile = os.path.join(args.data_path, args.dataset.replace('-', '_'), 'datasetInfo.pkl')
    if os.path.exists(pklFile):
        datasetInfo = pkl.load(open(pklFile, 'rb')) 
    else:
        args.addInverseRelation = False
        datasetInfo = prepareData(args)
        pkl.dump(datasetInfo, open(pklFile, "wb"))
    print('==> finish loading dataset')

    nentity       = datasetInfo['nentity']
    nrelation     = datasetInfo['nrelation']
    sample_ratio  = args.sample_ratio
    num_starts    = args.num_starts
    savePath      = args.data_path
    ori_train_triples = datasetInfo['train_triples']
    train_triples = []
    new_entity_dict = None

    if 'ogb' in args.dataset:
        if 'biokg' in args.dataset:
            entity_dict = datasetInfo['entity_dict']
            new_entity_dict = {}
            for idx in tqdm(range(len(ori_train_triples['head']))):
                head, relation, tail = ori_train_triples['head'][idx], ori_train_triples['relation'][idx], ori_train_triples['tail'][idx]
                head_type, tail_type = ori_train_triples['head_type'][idx], ori_train_triples['tail_type'][idx]
                head = head + entity_dict[head_type][0]
                tail = tail + entity_dict[tail_type][0]
                
                new_entity_dict[head] = head_type
                new_entity_dict[tail] = tail_type

                if relation < nrelation:
                    train_triples.append((head, relation, tail))
        
        elif 'wikikg2' in args.dataset:
            for idx in tqdm(range(len(ori_train_triples['head']))):
                head, relation, tail = ori_train_triples['head'][idx], ori_train_triples['relation'][idx], ori_train_triples['tail'][idx]
                if relation < nrelation:
                    train_triples.append((head, relation, tail))

    else:
        for head, relation, tail in tqdm(ori_train_triples):
            if relation < nrelation:
                train_triples.append((head, relation, tail))

    # clean cache
    del datasetInfo

    # generating the sampled KG
    generator = KGGenerator_randomWalks(train_triples, sample_ratio, num_starts, savePath, entity_dict=new_entity_dict)
