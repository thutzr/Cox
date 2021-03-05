import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pycox
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torch
import torchtuples as tt
from pycox.datasets import metabric
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH
np.random.seed(1234)
_ = torch.manual_seed(123)
import warnings
warnings.filterwarnings('ignore')

torch.multiprocessing.set_start_method('spawn')
pool = torch.multiprocessing.Pool(processes=100)

class WeightedConformalPrediction():
    def __init__(self,df,train_frac=0.6,calibration_frac=0.2,num_nodes=[32,32],
                 out_features=1,batch_norm=True,
                 batch_size=256,dropout=0.1,output_bias=False,
                epochs = 512, callbacks = [tt.callbacks.EarlyStopping()],
                 verbose = True,classification_model='XGBoost',
                 percentile = 0.95,epsilon=0.01):
        self.df = df
        self.p_t = len(self.df[self.df['event']==1])/len(self.df)
        self.train_frac = train_frac
        self.cali_frac = calibration_frac
        self.num_nodes = num_nodes
        self.out_features = out_features
        self.batch_norm = batch_norm
        self.batch_size = batch_size
        self.dropout = dropout
        self.output_bias = output_bias
        self.epochs = epochs
        self.callbacks = callbacks
        self.verbose = verbose
        self.clf_model = classification_model
        self.epsilon = epsilon
        self.percentile = percentile
        self.V = None
        self.W = None
        self.p_hat = None
        self.T_h = None
        self.x_mapper = None
        self.get_target = lambda df: (df['duration'].values, df['event'].values)
        self.bh = None
        self.model = None
        
    # 划分数据，输入原始数据，选择划分的比例，输出训练集验证集和calibration set
    def split_data(self):
        self.Z_tr = self.df.sample(frac=self.train_frac)
        self.df = self.df.drop(self.Z_tr.index)
        self.Z_ca = self.df.sample(frac=self.cali_frac/(1-self.train_frac))
        self.df = self.df.drop(self.Z_ca.index)
        self.Z_val = self.df
    
    def standardize(self):
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
        cols_leave = ['x4', 'x5', 'x6', 'x7']

        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]

        self.x_mapper = DataFrameMapper(standardize + leave)
        
        self.x_train = self.x_mapper.fit_transform(self.Z_tr).astype('float32')
        self.x_val = self.x_mapper.transform(self.Z_val).astype('float32')
        self.x_ca = self.x_mapper.transform(self.Z_ca).astype('float32')
        
        
        self.y_train = self.get_target(self.Z_tr)
        self.y_val = self.get_target(self.Z_val)
        self.durations_ca, self.events_ca = self.get_target(self.Z_ca)
        self.val = self.x_val, self.y_val
        self.in_features = self.x_train.shape[1]
        
    def preprocessing(self):
        self.split_data()
        self.standardize()

    def run_preprocessing(self):
        if self.x_mapper == None:
            self.preprocessing()
        
    def neural_network_cox(self):
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.1),

            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.1),

            torch.nn.Linear(32, self.out_features)
        )
        self.model = CoxPH(self.net, torch.optim.Adam)
        self.model.fit(self.x_train, self.y_train, self.batch_size, self.epochs, self.callbacks, self.verbose,
                val_data=self.val, val_batch_size=self.batch_size)
        
    def find_baseline_hazard_non_zero_idx(self):
        if self.model == None:
            self.neural_network_cox()
        self.baseline_hazards = self.model.compute_baseline_hazards()
        self.non_zero_idx = self.baseline_hazards[self.baseline_hazards>0].index[1] # 计算第一个非零元素的索引
        self.bh = self.baseline_hazards.loc[self.non_zero_idx]
        
    def compute_nonconformal_score_single(self,x,t):
        R = self.Z_tr[self.Z_tr['duration']>=t] # 找到at risk的人的covariates
        if len(R) == 0: # 如果没找到at risk的人就跳过
            return None
        x_R = self.x_mapper.transform(R).astype('float32')
        ch_r = self.model.predict_cumulative_hazards(x_R)
        exp_g_r = ch_r.loc[self.non_zero_idx]/self.bh
        return exp_g_r
    
    # 计算nonconformal score的函数，给定一个预测hazard的模型，training set
    # 和calibration set以及base hazard，输出结果
    def compute_nonconformal_score(self):
        # print('WCP:compute nonconformal score')
        if self.bh == None:
            self.find_baseline_hazard_non_zero_idx()
        Z_ca_1 = self.Z_ca[self.Z_ca['event']==1] # calibration set中发病的样本
        x_ca = self.x_mapper.transform(Z_ca_1).astype('float32')
        durations_test_1, events_test_1 = self.get_target(Z_ca_1)
        cumulative_hazards = self.model.predict_cumulative_hazards(x_ca)
        exp_g = cumulative_hazards.loc[self.non_zero_idx].div(self.bh)
        # input_ = [(x_ca[i],durations_test_1[i]) for i in range(len(x_ca))]
        # results = pool.starmap(self.compute_nonconformal_score_single,input_)
        self.V = []
        for i in range(len(x_ca)): # nonconformal score
            print('[{i}]')
            exp_g_r = self.compute_nonconformal_score_single(x_ca[i],durations_test_1[i])
            if exp_g_r is None:
                self.V.append(np.inf)
            else:
        # self.V = [np.log(exp_g[i])-np.log(np.sum(results[i])) for i in range(len(results))]
                self.V.append(np.log(exp_g[i])-np.log(np.sum(exp_g_r)))
        self.V = np.array(self.V+[np.inf])
        
    # 计算weight的函数，输入traning set, calibration set以及一个用来估计P(T=1|X=x)的分类模型
    def compute_weight(self):
        # print('WCP:compute weight')
        Z_ca_1 = self.Z_ca[self.Z_ca['event']==1]
        X_tr = self.x_train
        X_ca = self.x_mapper.transform(Z_ca_1).astype('float32')
        C_tr = self.Z_tr.iloc[:,-1] # training set的event,用于之后训练分类模型
        # 根据输入选择分类模型
        if self.clf_model == 'RF':
            from sklearn.ensemble import RandomForestClassifier
            self.clf = RandomForestClassifier(max_depth=2,random_state=0)
        elif self.clf_model == 'LR':
            from sklearn.linear_model import LogisticRegression
            self.clf = LogisticRegression(random_state=0)
        elif self.clf_model == 'XGBoost':
            import xgboost as xgb
            self.clf = xgb.XGBClassifier()
        self.clf.fit(X_tr,C_tr) # 训练分类模型
        p_predict = self.clf.predict_proba(X_ca)[:,1] # 预测p_hat
        self.W = np.divide(self.p_t,p_predict) # 估计w_hat
    
    def run_compute_nonconformal_score(self):
        if self.V == None:
            self.compute_nonconformal_score()
        
    def run_conpute_weight(self):
        if self.W == None:
            self.compute_weight()

    # 计算normalized weight,输入计算的weight，test point，训练过的分类模型
    def compute_normalized_weight(self,x):
        '''
        x: test point
        '''
        # print('WCP:compute normalized weight')
        p_predict = self.clf.predict_proba(x)[0,1] # 预测test point对应的T=1的概率
        w_predict = self.p_t/p_predict # 估计p_hat
        normalize_term = np.sum(self.W)+w_predict 
        p_hat = [i/normalize_term for i in self.W] # 计算所有病人的p_hat
        p_inf = w_predict/normalize_term # 计算无穷点的weight

        self.p_hat = np.array(p_hat+[p_inf])
 
    # 计算对应的置信区间，输入nonconformal score, normalized weight p_hat, p_inf,以及指定的percentile
    def compute_quantile(self,x,t):
        ch = self.model.predict_cumulative_hazards(x)
        exp_g_x = ch.loc[self.non_zero_idx]/self.bh
        exp_g_x_r = self.compute_nonconformal_score_single(x,t)
        if exp_g_x_r is None:
            return 1
        V_x = np.log(exp_g_x)-np.log(np.sum(exp_g_x_r))
        p_hat_leave = self.p_hat[self.V<=V_x[0]]
        return sum(p_hat_leave)
    
    def weighted_conformal_prediction(self,x):
        self.compute_normalized_weight(x)
        if self.p_hat[-1] > self.percentile:
            self.T_h = np.inf
        # print('WCP:weighted conformal prediction')
        else:
            quantile = 0
            t_l = 0
            t_h = 10
            # print('New Point Start')
            while (quantile<self.percentile):
                t_h = t_h + t_h*(self.percentile-quantile)
                # print(t_h*(self.percentile-quantile))
                
                quantile = self.compute_quantile(x,t_h)
                # print(quantile,t_h)
            quantile_ = quantile
            # t_l = t_h-10
            # flag = 0
            # while (quantile-self.percentile>self.epsilon):
            #     quantile = self.compute_quantile(x,t_l)
            #     print(quantile,t_l)
            #     if quantile < self.percentile:
            #         flag = 1
            #         break
            #     t_l = t_l - 10
            # self.T_h = t_l + flag*10
            return t_h
                
#         while (quantile<self.percentile):
#             t = (t_l+t_h)/2
#             quantile = self.compute_quantile(x,t)
#             print(quantile,t,t_l,t_h)
#             if (quantile >= self.percentile) and (abs(quantile-self.percentile)>self.epsilon):
#                 t_h = t
#                 t = (t_h+t_l)/2
#             else:
#                 t_l = t
#                 t = (t_h+t_l)/2
#             self.T_h = t

    def run_training_step(self):
        print('Preprocessing')
        self.run_preprocessing()
        print('Nonconformal Score')
        self.run_compute_nonconformal_score()
        print('Weight')
        self.run_conpute_weight()
            
    def get_T(self,x,t_h):
        t_h = self.weighted_conformal_prediction(x)
        return t_h
    
    def get_nonconformal_score_of_calibration(self):
        if self.V is None:
            self.compute_nonconformal_score()
        return self.V
    
    def get_weight(self):
        if self.W is None:
            self.compute_weight()
        return self.W
    
    def get_normalized_weight(self,x):
        if self.p_hat is None:
            self.compute_normalized_weight(x)
        return self.p_hat