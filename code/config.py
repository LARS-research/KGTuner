from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import os.path as osp
import numpy as np

# optimal model configs
global_cfg = argparse.Namespace()

'''
    TransE
'''
TransE_cfg = argparse.Namespace()
TransE_cfg.eval_params = {}
# 0.233 / 0.032 / 0.399 / 0.542
TransE_cfg.eval_params['wn18rr']       = {'n_neg': 128, 'regularizer': 'FRO', 'loss_function': 'BCE_adv', 'optimizer': 'adam', 'initializer': 'normal', 'shareInverseRelation': True, 'batch_size': 512, 'dim': 1000, 'lr': 0.00010262505431952937, 'regu_weight': 0.00041984443400820497, 'advs': 1.1461553831408073, 'dropoutRate': 0.003213409944382972, 'gamma': 3.5012462537188394, 'filter_falseNegative': True, 'label_smooth': 0.0, 'training_mode': 'negativeSampling', 'embedding_range': 0.01}
# 0.327 / 0.228 / 0.369 / 0.522
TransE_cfg.eval_params['FB15k_237']    = {'n_neg': 512, 'regularizer': 'FRO', 'loss_function': 'BCE_adv', 'optimizer': 'adam', 'initializer': 'xavier_norm', 'shareInverseRelation': True, 'batch_size': 512, 'dim': 1000, 'lr': 0.0002662759850681326, 'regu_weight': 0.0021635074747949298, 'advs': 1.9932647233325609, 'dropoutRate': 0.029648510035959984, 'gamma': 6.761775973293524, 'filter_falseNegative': True, 'label_smooth': 0.0, 'training_mode': 'negativeSampling', 'embedding_range': 0.01}
# Val MRR: 0.7787, Test MRR: 0.7781
TransE_cfg.eval_params['ogbl-biokg']   = {'n_neg': 128, 'regularizer': 'NUC', 'loss_function': 'CE', 'optimizer': 'adam', 'initializer': 'xavier_norm', 'shareInverseRelation': True, 'batch_size': 1024, 'dim': 2000, 'lr': 0.0001247365706688606, 'regu_weight': 0.006991362165987843, 'advs': 0, 'dropoutRate': 0.0006245731969567569, 'gamma': 7.6020531522274775, 'filter_falseNegative': True, 'label_smooth': 0.0, 'training_mode': 'negativeSampling', 'embedding_range': 0.01}
# Val MRR: 0.4932, Test MRR: 0.4739
TransE_cfg.eval_params['ogbl-wikikg2'] = {'n_neg': 128, 'regularizer': 'FRO', 'loss_function': 'CE', 'optimizer': 'adam', 'initializer': 'xavier_norm', 'shareInverseRelation': True, 'batch_size': 1024, 'dim': 100, 'lr': 0.000605988663916015, 'regu_weight': 1.5697088253247655e-05, 'advs': 0, 'dropoutRate': 0.016249266103338267, 'gamma': 21.054702009795296, 'filter_falseNegative': True, 'label_smooth': 0.0, 'embedding_range': 0.01, 'training_mode': 'negativeSampling'}

'''
    RotatE
'''
RotatE_cfg = argparse.Namespace()
RotatE_cfg.eval_params  = {}
# 0.480 / 0.427 / 0.501 / 0.582
RotatE_cfg.eval_params['wn18rr']       = {'n_neg': 2048, 'regularizer': 'FRO', 'loss_function': 'BCE_adv', 'optimizer': 'adam', 'initializer': 'normal', 'shareInverseRelation': True, 'batch_size': 512, 'dim': 1000, 'lr': 0.0012429089061604538, 'regu_weight': 5.137481843072869e-08, 'advs': 1.6606758402088124, 'dropoutRate': 0.004029530870438158, 'gamma': 3.780226406155406, 'filter_falseNegative': True, 'label_smooth': 0.0, 'embedding_range': 0.01, 'training_mode': 'negativeSampling'}
# 0.338 / 0.243 / 0.373 / 0.527
RotatE_cfg.eval_params['FB15k_237']    = {'n_neg': 128, 'regularizer': 'NUC', 'loss_function': 'BCE_adv', 'optimizer': 'adam', 'initializer': 'normal', 'shareInverseRelation': True, 'batch_size': 1024, 'dim': 2000, 'lr': 0.000589318941329376, 'regu_weight': 0.0002993899377697545, 'advs': 1.1210185387197085, 'dropoutRate': 0.015556885937495324, 'gamma': 14.462063563334059, 'filter_falseNegative': True, 'label_smooth': 0.0, 'training_mode': 'negativeSampling', 'embedding_range': 0.01}
# 0.8013 / 0.7226 / 0.8591 / 0.9420
RotatE_cfg.eval_params['ogbl-biokg']   = {'n_neg': 128, 'regularizer': 'DURA', 'loss_function': 'BCE_adv', 'optimizer': 'adam', 'initializer': 'normal', 'shareInverseRelation': True, 'batch_size': 1024, 'dim': 2000, 'lr': 0.0001119432728277056, 'regu_weight': 1.0992447497381416e-06, 'advs': 1.947248336918954, 'dropoutRate': 0.004228125242677383, 'gamma': 18.34106819763706, 'filter_falseNegative': True, 'label_smooth': 0.0, 'training_mode': 'negativeSampling', 'embedding_range': 0.01}
# 0.294 / 0.259 / 0.301 / 0.352
RotatE_cfg.eval_params['ogbl-wikikg2'] = {'n_neg': 32, 'regularizer': 'DURA', 'loss_function': 'CE', 'optimizer': 'adam', 'initializer': 'xavier_norm', 'shareInverseRelation': True, 'dim': 100, 'batch_size': 1024, 'lr': 0.040764216549273646, 'regu_weight': 0.008105636056545469, 'advs': 0, 'dropoutRate': 0.07494152163959279, 'gamma': 23.943425450406643, 'filter_falseNegative': True, 'label_smooth': 0.0, 'training_mode': 'negativeSampling', 'embedding_range': 0.01}

'''
    ComplEx
'''
ComplEx_cfg = argparse.Namespace()
ComplEx_cfg.eval_params  = {}
# 0.484 / 0.440 / 0.506 / 0.562
ComplEx_cfg.eval_params['wn18rr']       = {'n_neg': 32, 'regularizer': 'NUC', 'loss_function': 'BCE_mean', 'optimizer': 'adam', 'initializer': 'xavier_uniform', 'shareInverseRelation': False, 'batch_size': 1024, 'dim': 2000, 'lr': 0.0006085705049997063, 'regu_weight': 0.001219459833168948, 'advs': 0, 'dropoutRate': 0.2824915454386786, 'gamma': 2.291224978169301, 'filter_falseNegative': True, 'label_smooth': 0.0, 'training_mode': 'negativeSampling', 'embedding_range': 0.01}
# 0.352 / 0.263 / 0.387 / 0.530
ComplEx_cfg.eval_params['FB15k_237']    = {'n_neg': 512, 'regularizer': 'DURA', 'loss_function': 'BCE_adv', 'optimizer': 'adam', 'initializer': 'uniform', 'shareInverseRelation': True, 'batch_size': 1024, 'dim': 2000, 'lr': 0.0009706433869179586, 'regu_weight': 0.009753922487127387, 'advs': 1.9300950027456198, 'dropoutRate': 0.22146007031450685, 'gamma': 13.057616683834388, 'filter_falseNegative': True, 'label_smooth': 0.0, 'training_mode': 'negativeSampling', 'embedding_range': 0.01}
# Val MRR: 0.8394 Test MRR: 0.8385
ComplEx_cfg.eval_params['ogbl-biokg']   = {'n_neg': 512, 'regularizer': 'NUC', 'loss_function': 'CE', 'optimizer': 'adam', 'initializer': 'uniform', 'shareInverseRelation': True, 'batch_size': 1024, 'dim': 2000, 'lr': 0.0018915542437953079, 'regu_weight': 0.0013824721781032021, 'advs': 0, 'dropoutRate': 0.016532927574070145, 'gamma': 12.902627789934513, 'filter_falseNegative': True, 'label_smooth': 0.0, 'embedding_range': 0.01, 'training_mode': 'negativeSampling'}
# Val MRR: 0.5099, Test MRR: 0.4942
ComplEx_cfg.eval_params['ogbl-wikikg2'] = {'advs': 0, 'batch_size': 1024, 'dim': 100, 'dropoutRate': 0.00991169665189582, 'embedding_range': 0.01, 'filter_falseNegative': True, 'gamma': 6, 'initializer': 'xavier_norm', 'label_smooth': 0.0, 'loss_function': 'CE', 'lr': 0.00013428252531877898, 'n_neg': 32, 'optimizer': 'adam', 'regu_weight': 9.58942573228664e-07, 'regularizer': 'DURA', 'shareInverseRelation': True, 'training_mode': 'negativeSampling'}


'''
    DistMult
'''
DistMult_cfg = argparse.Namespace()
DistMult_cfg.eval_params  = {}
# 0.453 / 0.407 / 0.468 / 0.548
DistMult_cfg.eval_params['wn18rr']       = {'n_neg': 128, 'regularizer': 'NUC', 'loss_function': 'BCE_adv', 'optimizer': 'adam', 'initializer': 'normal', 'shareInverseRelation': True, 'batch_size': 1024, 'dim': 2000, 'lr': 0.004587034430717722, 'regu_weight': 0.009581702887741274, 'advs': 1.4156032814982573, 'dropoutRate': 0.29139837338726554, 'gamma': 12.881285331877564, 'filter_falseNegative': True, 'label_smooth': 0.0, 'training_mode': 'negativeSampling', 'embedding_range': 0.01}
# 0.345 / 0.254 / 0.377 / 0.527
DistMult_cfg.eval_params['FB15k_237']    = {'n_neg': 0, 'regularizer': 'NUC', 'loss_function': 'CE', 'optimizer': 'adam', 'initializer': 'xavier_uniform', 'shareInverseRelation': True, 'batch_size': 1024, 'dim': 1000, 'lr': 0.0004918908001373764, 'regu_weight': 0.002133680066894589, 'advs': 0, 'dropoutRate': 0.298886917379427, 'gamma': 2.9095475279141327, 'filter_falseNegative': True, 'label_smooth': 0.0, 'training_mode': 'kVsAll', 'embedding_range': 0.01}
# Val MRR: 0.8245 Test MRR: 0.8241
DistMult_cfg.eval_params['ogbl-biokg']   = {'n_neg': 512, 'regularizer': 'NUC', 'loss_function': 'CE', 'optimizer': 'adam', 'initializer': 'xavier_uniform', 'shareInverseRelation': True, 'batch_size': 1024, 'dim': 1000, 'lr': 0.0012599508388144562, 'regu_weight': 1.2043959875795455e-06, 'advs': 0, 'dropoutRate': 0.0035481935568574038, 'gamma': 11.826213596876292, 'filter_falseNegative': True, 'label_smooth': 0.0, 'embedding_range': 0.01, 'training_mode': 'negativeSampling'}
# Val MRR: 0.5004 Test MRR: 0.4837
DistMult_cfg.eval_params['ogbl-wikikg2'] = {'advs': 0, 'batch_size': 1024, 'dim': 100, 'dropoutRate': 0.005360432180037816, 'embedding_range': 0.01, 'filter_falseNegative': True, 'gamma': 6, 'initializer': 'xavier_norm', 'label_smooth': 0.0, 'loss_function': 'CE', 'lr': 0.00019846224510251668, 'n_neg': 32, 'optimizer': 'adam', 'regu_weight': 2.4386540768609533e-08, 'regularizer': 'DURA', 'shareInverseRelation': True, 'training_mode': 'negativeSampling'}

'''
    TuckER
'''
TuckER_cfg = argparse.Namespace()
TuckER_cfg.eval_params  = {}
# 0.480 / 0.437 / 0.500 / 0.557
TuckER_cfg.eval_params['wn18rr']       = {'n_neg': 128, 'regularizer': 'DURA', 'loss_function': 'BCE_adv', 'optimizer': 'adam', 'initializer': 'xavier_uniform', 'shareInverseRelation': True, 'batch_size': 512, 'dim': 200, 'lr': 0.0026031746116015153, 'regu_weight': 0.00222505986437916, 'advs': 1.9447536069105926, 'dropoutRate': 0.00020952469652523175, 'gamma': 12.977283472746866, 'filter_falseNegative': True, 'label_smooth': 0.0, 'training_mode': 'negativeSampling', 'embedding_range': 0.01, 'tucker_drop1': 0.0, 'tucker_drop2': 0.0}
# 0.347 / 0.255 / 0.382 / 0.534
TuckER_cfg.eval_params['FB15k_237']    = {'n_neg': 2048, 'regularizer': 'DURA', 'loss_function': 'BCE_adv', 'optimizer': 'adam', 'initializer': 'normal', 'shareInverseRelation': True, 'batch_size': 1024, 'dim': 500, 'lr': 0.0003354924877357634, 'regu_weight': 0.0002667563092274356, 'advs': 1.955898058210909, 'dropoutRate': 0.018186237512921432, 'gamma': 13.517054518756952, 'filter_falseNegative': True, 'label_smooth': 0.0, 'embedding_range': 0.01, 'training_mode': 'negativeSampling', 'tucker_drop1': 0.0, 'tucker_drop2': 0.0}

'''
    ConvE
'''
ConvE_cfg = argparse.Namespace()
ConvE_cfg.eval_params  = {}
# 0.437 / 0.399 / 0.449 / 0.515
ConvE_cfg.eval_params['wn18rr']       = {'n_neg': 512, 'regularizer': 'DURA', 'loss_function': 'BCE_adv', 'optimizer': 'adam', 'initializer': 'xavier_uniform', 'shareInverseRelation': True, 'batch_size': 512, 'dim': 1000, 'lr': 0.0006889495169935946, 'regu_weight': 0.009798840758912599, 'advs': 0.7808368819289935, 'dropoutRate': 0.023066362929213213, 'gamma': 12.163000923835527, 'filter_falseNegative': True, 'label_smooth': 0.0, 'conve_drop1': 0.0, 'conve_drop2': 0.0, 'embedding_range': 0.01, 'training_mode': 'negativeSampling'}
# 0.335 / 0.242 / 0.368 / 0.523
ConvE_cfg.eval_params['FB15k_237']    = {'n_neg': 512, 'regularizer': 'DURA', 'loss_function': 'BCE_sum', 'optimizer': 'adam', 'initializer': 'normal', 'shareInverseRelation': True, 'batch_size': 1024, 'dim': 500, 'lr': 0.00020996286973689828, 'regu_weight': 0.006426220156393892, 'advs': 0, 'dropoutRate': 0.07913995369005729, 'gamma': 14.527668521455467, 'filter_falseNegative': True, 'label_smooth': 0.0, 'embedding_range': 0.01, 'training_mode': 'negativeSampling', 'conve_drop1': 0.0, 'conve_drop2': 0.0}

'''
    RESCAL
'''
RESCAL_cfg = argparse.Namespace()
RESCAL_cfg.eval_params  = {}
# 0.479 / 0.436 / 0.496 / 0.557
RESCAL_cfg.eval_params['wn18rr']       = {'n_neg': 128, 'regularizer': 'DURA', 'loss_function': 'BCE_mean', 'optimizer': 'adam', 'initializer': 'uniform', 'shareInverseRelation': True, 'batch_size': 512, 'dim': 1000, 'lr': 0.001736879142796609, 'regu_weight': 0.0017699373743384633, 'advs': 0, 'dropoutRate': 0.003511500208180762, 'gamma': 2.4127211860255775, 'filter_falseNegative': True, 'label_smooth': 0.0, 'training_mode': 'negativeSampling', 'embedding_range': 0.01}
# 0.357 / 0.268 / 0.390 / 0.535
RESCAL_cfg.eval_params['FB15k_237']    = {'n_neg': 2048, 'regularizer': 'DURA', 'loss_function': 'CE', 'optimizer': 'adam', 'initializer': 'xavier_norm', 'shareInverseRelation': True, 'batch_size': 512, 'dim': 500, 'lr': 0.0009307704780775203, 'regu_weight': 0.008346484669054724, 'advs': 0, 'dropoutRate': 0.010462601679153393, 'gamma': 4.17638032844613, 'filter_falseNegative': True, 'label_smooth': 0.0, 'training_mode': 'negativeSampling', 'embedding_range': 0.01}


'''
    AutoSF
'''
AutoSF_cfg = argparse.Namespace()
AutoSF_cfg.eval_params = {}
# Val MRR: 0.8361, Test MRR: 0.8354
AutoSF_cfg.eval_params['ogbl-biokg']   = {'n_neg': 512, 'regularizer': 'NUC', 'loss_function': 'CE', 'optimizer': 'adam', 'initializer': 'uniform', 'shareInverseRelation': True, 'batch_size': 1024, 'dim': 2000, 'lr': 0.0018915542437953079, 'regu_weight': 0.0013824721781032021, 'advs': 0, 'dropoutRate': 0.016532927574070145, 'gamma': 12.902627789934513, 'filter_falseNegative': True, 'label_smooth': 0.0, 'embedding_range': 0.01, 'training_mode': 'negativeSampling'}
# Val MRR: 0.5397, Test MRR: 0.5222
AutoSF_cfg.eval_params['ogbl-wikikg2'] = {'n_neg': 2048, 'regularizer': 'None', 'loss_function': 'CE', 'optimizer': 'adam', 'initializer': 'xavier_norm', 'shareInverseRelation': False, 'batch_size': 1024, 'dim': 100, 'lr': 0.0074032731130446, 'regu_weight': 0, 'advs': 0, 'dropoutRate': 0.02602294263336368, 'gamma': 18.914835894398028, 'filter_falseNegative': True, 'label_smooth': 0.0, 'training_mode': 'negativeSampling', 'embedding_range': 0.01}

global_cfg.TransE_cfg   = TransE_cfg
global_cfg.DistMult_cfg = DistMult_cfg
global_cfg.RotatE_cfg   = RotatE_cfg
global_cfg.ComplEx_cfg  = ComplEx_cfg
global_cfg.RESCAL_cfg   = RESCAL_cfg
global_cfg.ConvE_cfg    = ConvE_cfg
global_cfg.TuckER_cfg   = TuckER_cfg
global_cfg.AutoSF_cfg   = AutoSF_cfg

# full space
global_cfg.full_space = {
        # full search space for batch_size and dim
        'batch_size':               ('choice', [512, 1024]), 
        'dim':                      ('choice', [1000, 2000]), 

        # discrete
        'n_neg':                    ('choice', ['32', '128', '512', '2048', 'kVsAll','1VsAll']), 
        'regularizer':              ('choice', ['FRO', 'NUC', 'DURA', 'None']),
        'loss_function':            ('choice', ['MR','BCE_mean','BCE_sum','BCE_adv','CE']),
        'initializer':              ('choice', ['uniform','xavier_norm', 'xavier_uniform', 'normal']), 
        
        # continous 
        'lr':                       ('uniform', (-4, -1)),
        'regu_weight':              ('uniform', (-8, -2)),
        'advs':                     ('uniform', (0.5, 2.0)),
        'dropoutRate':              ('uniform', (0.0, 0.3)),
        'gamma':                    ('uniform', [1, 24]),  

        # fixed
        'optimizer':                ('choice', ['adam']), 
        'shareInverseRelation':     ('choice', [True]),
    }

# reduced space
global_cfg.reduced_space = {
        # discrete
        'n_neg':                    ('choice', ['32', '128', '512', '2048', 'kVsAll','1VsAll']), 
        'regularizer':              ('choice', ['FRO', 'NUC', 'DURA', 'None']),
        'loss_function':            ('choice', ['MR','BCE_mean','BCE_sum','BCE_adv','CE']),
        'initializer':              ('choice', ['uniform','xavier_norm', 'xavier_uniform', 'normal']), 
        
        # continous 
        'lr':                       ('uniform', (-4, -1)),
        'regu_weight':              ('uniform', (-8, -2)),
        'advs':                     ('uniform', (0.5, 2.0)),
        'dropoutRate':              ('uniform', (0.0, 0.3)),
        'gamma':                    ('uniform', [1, 24]),  

        # fixed
        'optimizer':                ('choice', ['adam']), 
        'shareInverseRelation':     ('choice', [True]),
        'batch_size':               ('choice', [128]), 
        'dim':                      ('choice', [100]),
}


'''
    Other training settings that are not covered in the search space
'''

training_strategy = argparse.Namespace()
training_strategy.filter_falseNegative = True
# True or False
training_strategy.double_entity_embedding = False
# True or False
training_strategy.double_relation_embedding = False
# 0.0 - 1.0
training_strategy.label_smooth = 0.0
# 0.0 - 1.0
training_strategy.embedding_range = 1e-2
# True or False: if False, subsampling_weight will be taken into loss calculation
training_strategy.uni_weight = False
# True or False
training_strategy.require_subsampling_weight    = True
# True or False
training_strategy.negative_adversarial_sampling = True

global_cfg.training_strategy = training_strategy
