import random
import copy
import numpy as np
from   sklearn import ensemble
from   scipy.stats import norm
from   tqdm import tqdm

class base_HPO():
    def __init__(self, kgeModelName, obj_function, HP_info, dataset_name=None, acq='maxMean', meta_feature=None, msg=None, model_type='single'):
        self.kgeModelName         = kgeModelName
        self.dataset_name         = dataset_name
        self.dataset_names        = []
        self.obj_function         = obj_function
        self.HP_info              = HP_info
        self.observed_config_dict = []
        self.observed_x           = []
        self.observed_y           = []
        self.pred_y               = []
        self.best                 = -1
        self.last_fit_observe_num = 0
        self.finetune_HP_list     = ['advs', 'gamma', 'dropoutRate', 'regu_weight']
        self.trained_flag         = False
        self.finetune_flag        = False
        self.meta_feature         = meta_feature
        self.acq                  = acq
        self.acq_dist             = {'EI':self.EI, 'UCB':self.UCB, 
                                    'max':self.maxMean, 'BORE':self.maxMean,
                                    'None':None}
        self.acq_function         = self.acq_dist[acq]
        self.model_type           = model_type
        self.summary              = dict()
        self.summary['acq']       = acq
        self.summary['msg']       = msg
        

    def getSummary(self):
        self.summary['observed_x'] = self.observed_x
        self.summary['observed_y'] = self.observed_y
        self.summary['pred_y']     = self.pred_y
        self.summary['observed_config_dict'] = self.observed_config_dict

        return self.summary

    def checkConfig(self, config):
        training_mode_flag = config['n_neg'] in ['1VsAll','kVsAll']
        if self.kgeModelName in ['TransE','RotatE','pRotatE'] and training_mode_flag:
            return 'FAIL'
        if config['loss_function'] in ['MR', 'BCE_adv'] and training_mode_flag:
            return 'FAIL'  
            
        # if not training_mode_flag and config['n_neg'] <= 0:
        #     return 'FAIL'  
        # if config['loss_function'] == 'MR' and config['gamma'] <= 0:
        #     return 'FAIL'  
        
        return 'PASS'

    def randomSampleOneConfig(self):
        tmp_config = {}
        for HP_name, info in self.HP_info.items():
            HP_type, HP_range = info[0], info[1]
            if HP_type == 'uniform':
                tmp_config[HP_name] = np.random.uniform(HP_range[0], HP_range[1])
            else:
                tmp_config[HP_name] = np.random.choice(HP_range, 1)[0]

        return tmp_config

    def randomSampleConfig(self, sample_num=1e4):
        candidate_config = []

        while len(candidate_config) < sample_num:
            tmp_config = self.randomSampleOneConfig()

            # TODO: check here
            if self.checkConfig(tmp_config) == 'PASS':
                candidate_config.append(tmp_config)

        return candidate_config

    def getCondidateConifg(self, sample_num=1e4, iterations=1e4):
        sample_num, iterations = int(sample_num), int(iterations)
        candidate_config = self.randomSampleConfig(sample_num=sample_num)
        if not self.trained_flag:
            return candidate_config

        # print(np.mean(self.getACQResults(candidate_config)))
        updateIteration = int(iterations / 10)
        for i in range(iterations):
            p = random.random()

            if p < 0.2:
                # P=20% random sample
                new_config = self.randomSampleOneConfig()
            elif p < 0.6:
                # P=40% to do mutation
                index      = random.randint(0, sample_num-1)
                config     = copy.deepcopy(candidate_config[index])
                new_config = self.mutation(config)
            else:
                # P=40% to do crossover
                index1, index2   = random.randint(0, sample_num-1), random.randint(0, sample_num-1)
                config1, config2 = copy.deepcopy(candidate_config[index1]), copy.deepcopy(candidate_config[index2])
                new_config       = self.crossover(config1, config2)

            candidate_config.append(new_config)

            if (i+1) % updateIteration == 0:
                acqScores          = self.getACQResults(candidate_config)
                topConfigIndex     = np.argsort(acqScores)[::-1][:sample_num]
                candidate_config   = list(np.array(candidate_config)[topConfigIndex])

        return candidate_config

    def getFinetunedCondidateConifg(self, sample_num=1e3, finetuneOtherHP=False):
        ''' finetune from observed configs '''
        candidate_config = copy.deepcopy(self.observed_config_dict)
        generated_finetune_config = []

        # increase dim/bs -> add combination 
        for cfg in candidate_config:
            for dim in self.HP_info['dim'][1][-2:]:
                for batch_size in self.HP_info['batch_size'][1][-2:]:
                    tmp_cfg = copy.deepcopy(cfg)
                    tmp_cfg['dim']        = dim
                    tmp_cfg['batch_size'] = batch_size

                    if tmp_cfg not in generated_finetune_config and tmp_cfg not in candidate_config:
                        generated_finetune_config.append(tmp_cfg)

        # optional: modify config for fintune
        if finetuneOtherHP:
            for cfg in candidate_config:
                for i in range(int(sample_num)):
                    tmp_cfg = copy.deepcopy(cfg)                
                    for HP_name in self.finetune_HP_list:
                        if HP_name in ['gamma', 'advs'] and tmp_cfg[HP_name] == 0:
                            continue
                        HP_range = self.HP_info[HP_name][1]
                        tmp_cfg[HP_name] = np.random.uniform(HP_range[0], HP_range[1])
                    
                    if tmp_cfg not in generated_finetune_config and tmp_cfg not in candidate_config:
                            generated_finetune_config.append(tmp_cfg)

        # print(f'==> len(generated_finetune_config): {len(generated_finetune_config)}')
        return generated_finetune_config

    def getACQResults(self, candidate_config):
        assert(self.trained_flag)

        candidate_config_X = self.changeConfigToArray(candidate_config)

        if type(self.meta_feature) is np.ndarray:
            feature_matrix     = np.tile(self.meta_feature, (candidate_config_X.shape[0], 1))
            candidate_config_X = np.concatenate((candidate_config_X, feature_matrix), axis=1)

        mu, std = self.predict_with_std(candidate_config_X)
        cfg_index, cfg_pred_y, acqScores = self.acq_function(mu=mu, std=std)

        return acqScores

    def mutation(self, config, times=1):
        HP_num = len(self.HP_info.keys())

        for i in range(times):
            HP_index = random.randint(0, HP_num-1)
            HP_name  = list(self.HP_info.keys())[HP_index]
            HP_type, HP_range = self.HP_info[HP_name] 

            if HP_type == 'uniform':
                config[HP_name] = np.random.uniform(HP_range[0], HP_range[1])
            else:
                config[HP_name] = np.random.choice(HP_range, 1)[0]

        return config

    def crossover(self, config1, config2):
        new_config = {}
        for HP_name in self.HP_info.keys():
            p = random.random()
            if p <= 0.5:
                new_config[HP_name] = config1[HP_name]
            else:
                new_config[HP_name] = config2[HP_name]

        return new_config

    def changeConfigToArray(self, config):
        cfg_array_list = []

        for cfg in config:
            tmp_array = []
            for HP_name, info in self.HP_info.items():
                HP_type, HP_range = info[0], info[1]

                if not isinstance(HP_range[0], str):
                    try:
                        tmp_array.append(float(cfg[HP_name]))
                    except:
                        tmp_array.append(float(HP_range[0]))
                
                else:
                    one_hot_array = [0 for i in range(len(HP_range))]
                    try:
                        one_hot_array[HP_range.index(cfg[HP_name])] = 1
                    except:
                        # print('[!] Error in changeConfigToArray(): ', HP_name, cfg[HP_name], cfg)  
                        one_hot_array[0] = 1  

                    tmp_array += one_hot_array
            
            cfg_array_list.append(np.array(tmp_array))
        
        cfg_np_array = np.array(cfg_array_list, dtype=float)
        return cfg_np_array

    def pretrain(self, config_list, mrr_list, dataset_names):
        ''' pretrain with this KG's records '''
        X = self.changeConfigToArray(config_list)
        X[np.where(np.isinf(X))] = 0
        Y = np.array(mrr_list)
        self.dataset_names = dataset_names # record dataset name for each observed (x, y) 

        if self.acq == 'BORE':
            Z = self.BORE_convert_batch_labels(Y)
            self.fit(X, Z, pretrain=True)
        else:
            self.fit(X, Y, pretrain=True)

        self.summary['pretrain_X'] = X
        self.summary['pretrain_Y'] = Y

        for idx in range(len(config_list)):
            # self.observed_config_dict.append(config_list[idx])
            if dataset_names[idx] == self.dataset_name:
                self.observed_config_dict.append(config_list[idx])
            self.observed_x.append(list(self.changeConfigToArray([config_list[idx]])[0]))
            self.observed_y.append(float(mrr_list[idx]))

    def pretrain_with_meta_feature(self, config_list, mrr_list, dataset_names, meta_feature, topNumToStore=0):
        ''' 
            config_list are of other datasets (can include this KG)
            Notes that X is concateaned with meta feature
        '''
        feature_matrix = np.array([meta_feature[name] for name in dataset_names])
        X = self.changeConfigToArray(config_list)
        # print(X.shape, feature_matrix.shape)
        X = np.concatenate((X, feature_matrix), axis=1)
        X[np.where(np.isinf(X))] = 0
        X[np.where(np.isnan(X))] = 0
        Y = np.array(mrr_list)
        self.dataset_names = dataset_names # record dataset name for each observed (x, y) 

        if self.acq == 'BORE':
            Z = self.BORE_convert_batch_labels(Y)
            # print(f'acq=BORE, \nX={X}\nY={Y}\nZ={Z}')
            self.fit(X, Z, pretrain=True)
        else:
            self.fit(X, Y, pretrain=True)
    
        self.summary['pretrain_X'] = X
        self.summary['pretrain_Y'] = Y

        # fine-tune top-k configures 
        if topNumToStore > 0:
            self.finetune_flag = True
            topNumidx = np.argsort(mrr_list)[::-1][:topNumToStore]
            for idx in topNumidx:
                self.observed_config_dict.append(config_list[idx])

        # utilize history data
        for idx in range(X.shape[0]):
            self.observed_x.append(list(X[idx]))
            self.observed_y.append(float(Y[idx]))


    def BORE_convert_batch_labels(self, Y, valid_index=None):
        ''' Y -> Z for a batch of datasets'''
        
        dataset_names = np.array(self.dataset_names)[valid_index] if valid_index != None else np.array(self.dataset_names)
        assert(Y.shape[0] == len(dataset_names)) 

        Z = [] 
        for data_name in sorted(set(dataset_names), key=list(dataset_names).index):
            indexs = np.where(dataset_names == data_name)[0]
            Z      += list(self.BORE_convert_label(Y[indexs]))

        return np.array(Z)

    def BORE_convert_label(self, Y, gamma=0.75):
        ''' Y -> Z for one dataset'''
        margin_y = sorted(list(Y))[int(Y.shape[0] * gamma)]
        Z        = np.zeros(Y.shape)
        Z[np.where(Y >= margin_y)[0]] = 1

        return Z
        
    def fit(self, X, Y, pretrain=False, valid_index=None):
        if valid_index != None: X, Y = X[valid_index], Y[valid_index]

        # before HPO starts
        if pretrain:
            try:
                # print(f'pretraining with X={X.shape}, Y={Y.shape}')
                if self.model_type == 'dual':
                    self.model.fit_shared_estimators(X, Y)
                else:
                    self.model.fit(X, Y)
                self.trained_flag = True
            except:
                print('Error in fit.pretrain, Y:', Y)
            return

        # in normal HPO pipeline
        try:
            if self.acq == 'BORE':
                Z = self.BORE_convert_batch_labels(Y, valid_index)
                if len(set(Z)) != 2: return # invalid training data (should be with 0/1 labels)
                
                if self.model_type == 'dual':
                    self.model.fit_task_estimators(X, Z)
                else:
                    self.model.fit(X, Z)
                
            else:
                if self.model_type == 'dual':
                    self.model.fit_task_estimators(X, Y)
                else:
                    self.model.fit(X, Y)

            self.trained_flag = True
            self.best = np.max(Y)
            self.last_fit_observe_num = Y.shape[0]

        except:
            print('Error in fit(), Y:', Y)
            return
        
            
    def predict(self, X):
        # check input
        X[np.where(np.isinf(X))] = 0
        X[np.where(np.isnan(X))] = 0

        try:
            pred = self.model.predict(X)
        except:
            pred = [-1 for i in range(X.shape[0])]
        return pred

    def predict_with_std(self, X):
        # check input
        X[np.where(np.isinf(X))] = 0
        X[np.where(np.isnan(X))] = 0

        if self.summary['surrogate'] == 'random forest':
            full_predict_results = np.zeros((len(self.model.estimators_), X.shape[0]))
            for idx, tree in enumerate(self.model.estimators_):
                if self.acq == 'BORE':
                    full_predict_results[idx] = tree.predict_proba(X)[:, 1]
                else:
                    full_predict_results[idx] = tree.predict(X)

            mu, std = np.mean(full_predict_results, axis=0), np.std(full_predict_results, axis=0)

        elif self.summary['surrogate'] == 'gradient boosting':
            full_predict_results = np.zeros((len(self.model.estimators_), X.shape[0]))
            for idx, tree in enumerate(self.model.estimators_):
                full_predict_results[idx] = tree[0].predict(X) # difference here

            mu, std = np.mean(full_predict_results, axis=0), np.std(full_predict_results, axis=0)
        
        elif self.summary['surrogate'] == 'ngboosting':
            dists = self.model.pred_dist(X)
            mu, std = dists.mean(), dists.scale

        else: 
            print('[Error] predict_with_std() is not supported for:', self.summary['surrogate'])
            exit()

        return mu, std
    
    def EI(self, mu, std, mask=None):
        gamma = (mu - self.best) / std
        EI_scores = (std * gamma * norm.cdf(gamma) + std * norm.pdf(gamma)).flatten()
    
        if mask != None:
            EI_scores[mask] = -100
        
        max_index = np.argmax(EI_scores)
        return max_index, mu[max_index], EI_scores

    def UCB(self, mu, std, tradeoff=2.56, mask=None):
        
        UCB_scores   = (mu + tradeoff * std).flatten()
        highStdIndex = np.where(std > 1.0)[0]
        lowMuIndex   = np.where(mu < self.best * 0.9)[0]
        
        if mask != None:
            mu[mask]         = -100
            UCB_scores[mask] = -100
        
        UCB_scores[highStdIndex] *= 1e-3
        UCB_scores[lowMuIndex]   *= 1e-3
        
        max_index = np.argmax(UCB_scores)
        return max_index, mu[max_index], UCB_scores

    def maxMean(self, mu, std=None, mask=None):
        if mask != None:
            mu[mask] = -100

        max_index = np.argmax(mu)
        return max_index, mu[max_index], mu

    def runTrials(self, maxTrials, sample_num, meta_feature=None):
        self.summary['sample_num'] = sample_num 
        self.meta_feature = meta_feature

        for trial in range(int(maxTrials)):
            
            # generate candidate configs
            if self.finetune_flag:
                # exploitation via finetuning top config
                candidate_config = self.getFinetunedCondidateConifg() 
                if len(candidate_config) == 0:
                    self.finetune_flag = False
                    candidate_config = self.getCondidateConifg(sample_num=sample_num) # exploration
            else:
                candidate_config = self.getCondidateConifg(sample_num=sample_num) # exploration
            
            # select config with best predicting result
            candidate_config_X = self.changeConfigToArray(candidate_config)
            if type(meta_feature) is np.ndarray:
                feature_matrix     = np.tile(meta_feature, (candidate_config_X.shape[0], 1))
                candidate_config_X = np.concatenate((candidate_config_X, feature_matrix), axis=1)
            if self.trained_flag:
                mu, std = self.predict_with_std(candidate_config_X)
                cfg_index, cfg_pred_y, _ = self.acq_function(mu=mu, std=std)
            else:
                cfg_index  = np.random.randint(sample_num)
                cfg_pred_y = -1
            next_config = candidate_config[cfg_index]

            # change config from dict to array
            if type(meta_feature) is np.ndarray:
                next_config_X = list(self.changeConfigToArray([next_config])[0]) + list(meta_feature)
            else:
                next_config_X = list(self.changeConfigToArray([next_config])[0])

            # run on obj_function (KGEModel training)
            trial_result = self.obj_function(copy.deepcopy(next_config))
            real_y = trial_result['loss']
            
            # record
            self.observed_x.append(next_config_X)
            self.observed_y.append(float(1-real_y))
            self.observed_config_dict.append(next_config)
            self.pred_y.append(cfg_pred_y)
            self.dataset_names.append(self.dataset_name)

            # fit surrogate
            if len(self.observed_y) > 1:
                X, Y = np.array(self.observed_x), np.array(self.observed_y)
                valid_index = np.where(Y > 0)
                X[np.where(np.isinf(X))] = 0
                X[np.where(np.isnan(X))] = 0
                # print(f'==> [{self.modelName}. train model]: X.shape={X.shape}, Y.shape={Y.shape}')
                self.fit(X, Y, valid_index=valid_index)

class RF_HPO(base_HPO):
    def __init__(self, kgeModelName, obj_function, HP_info, dataset_name=None, acq='maxMean', meta_feature=None, msg=None, model_type='single'):        
        base_HPO.__init__(self, kgeModelName, obj_function, HP_info, dataset_name, acq, meta_feature, msg, model_type)
        params = {'n_estimators': 200}
        
        if acq == 'BORE':
            # classifier for BORE
            self.model = ensemble.RandomForestClassifier(**params)
        else:
            # regressor for other acquisition functions
            self.model = ensemble.RandomForestRegressor(**params)

        self.modelName = 'random forest'
        self.summary['surrogate'] = self.modelName
        # print(f'==> surrogate={self.modelName}, acq={acq}')
