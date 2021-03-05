import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pycox
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn_pandas import DataFrameMapper
import torch
import torchtuples as tt
from pycox.evaluation import EvalSurv
from pycox.models import PCHazard
import warnings
warnings.filterwarnings('ignore')
import joblib

class WeightedConformalPrediction():
    def __init__(self,df_train,train_frac = 0.8,num_nodes=[32,32],
                 out_features=1,batch_norm=True,
                 batch_size=128,dropout=0.1,output_bias=False,
                epochs = 512, callbacks = [tt.callbacks.EarlyStopping()],
                 verbose = True,classification_model='LR',
                 percentile = 0.95,epsilon=0.01):
        self.df_train = df_train
        # self.p_t = len(self.df[self.df['event']==1])/len(self.df)
        self.train_frac = train_frac
        # self.cali_frac = calibration_frac
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
        random_idx = np.random.permutation(range(len(self.df_train)))
        train_idx = random_idx[:int(len(self.df_train)*self.train_frac)]
        val_idx = random_idx[int(len(self.df_train)*self.train_frac):]
        self.Z_tr = self.df_train.iloc[train_idx,:]
        self.Z_val = self.df_train.iloc[val_idx,:]
    
    def standardize(self):
        cols_standardize = ['x0', 'x1', 'x2']
        # cols_leave = ['x4', 'x5', 'x6', 'x7']

        standardize = [([col], StandardScaler()) for col in cols_standardize]
        # leave = [(col, None) for col in cols_leave]
        polyfeature = [([col], PolynomialFeatures()) for col in cols_standardize]
        self.x_mapper = DataFrameMapper(standardize+polyfeature)
        
        self.x_train = self.x_mapper.fit_transform(self.Z_tr).astype('float32')
        self.x_val = self.x_mapper.transform(self.Z_val).astype('float32')
        # self.x_ca = self.x_mapper.transform(self.Z_ca).astype('float32')
        
        
        self.y_train = self.get_target(self.Z_tr)
        self.y_val = self.get_target(self.Z_val)
        # self.durations_ca, self.events_ca = self.get_target(self.Z_ca)
        self.val = self.x_val, self.y_val
        self.in_features = self.x_train.shape[1]
        
    def preprocessing(self):
        self.split_data()
        self.standardize()

    def run_preprocessing(self):
        if self.x_mapper == None:
            self.preprocessing()
        
    def neural_network_cox(self):
        self.net = tt.practical.MLPVanilla(self.in_features,self.num_nodes,self.out_features,self.batch_norm,self.dropout)
        self.model = PCHazard(self.net, torch.optim.Adam)
        self.model.fit(self.x_train, self.y_train, self.batch_size, self.epochs, self.callbacks, self.verbose,
                val_data=self.val, val_batch_size=self.batch_size)
        
    def find_baseline_hazard_non_zero_idx(self):
        if self.model == None:
            self.neural_network_cox()
        self.baseline_hazards = self.model.compute_baseline_hazards()
        self.non_zero_idx = self.baseline_hazards[self.baseline_hazards>0].index[1] # 计算第一个非零元素的索引
        self.bh = self.baseline_hazards.loc[self.non_zero_idx]
        
    def compute_nonconformal_score_single(self,t):
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
        self.V = list()
        for i in range(len(x_ca)): # nonconformal score
            exp_g_r = self.compute_nonconformal_score_single(durations_test_1[i])
            if exp_g_r is None:
                self.V.append(np.inf)
            else:
                self.V.append(np.log(exp_g[i])-np.log(np.sum(exp_g_r)+exp_g[i]))
        print('[Mean]\t%.2f\t [Std.]\t %.2f\t[Max]\t%.2f\t[Min]\t%.2f'%(np.mean(self.V),np.std(self.V),np.max(self.V),np.min(self.V)))
        self.V = np.array(self.V+[np.inf])
        
    # 计算weight的函数，输入traning set, calibration set以及一个用来估计P(T=1|X=x)的分类模型
    def compute_weight(self):
        # print('WCP:compute weight')
        Z_ca_1 = self.Z_ca[self.Z_ca['event']==1]
        X_tr = self.x_train
        X_ca = self.x_mapper.transform(Z_ca_1).astype('float32')
        C_tr = self.Z_tr['event'] # training set的event,用于之后训练分类模型
        # 根据输入选择分类模型
        if self.clf_model == 'RF':
            from sklearn.ensemble import RandomForestClassifier
            self.clf = RandomForestClassifier(max_depth=6,random_state=0)
        elif self.clf_model == 'LR':
            from sklearn.linear_model import LogisticRegression
            self.clf = LogisticRegression(random_state=0)
        elif self.clf_model == 'XGBoost':
            import xgboost as xgb
            self.clf = xgb.XGBClassifier()
        self.clf.fit(X_tr,C_tr) # 训练分类模型
        p_predict = self.clf.predict_proba(X_ca)[:,1] # 预测p_hat
        p_predict[p_predict<0.1] = 0.1
        p_predict[p_predict>0.9] = 0.9
        print(np.max(p_predict),np.min(p_predict))
        self.W = np.divide(1-p_predict,p_predict) # 估计w_hat
    
    def run_compute_nonconformal_score(self):
        if self.V == None:
            self.compute_nonconformal_score()
        else:
            pass 
        
    def run_conpute_weight(self):
        if self.W == None:
            self.compute_weight()
        else:
            pass

    # 计算normalized weight,输入计算的weight，test point，训练过的分类模型
    def compute_normalized_weight(self,x):
        '''
        x: test point
        '''
        # print('WCP:compute normalized weight')
        p_predict = self.clf.predict_proba(x)[0,1] # 预测test point对应的T=1的概率
        w_predict = self.p_t/p_predict # 估计p_hat
        normalize_term = np.sum(self.W)+w_predict 
        p_hat = self.W/normalize_term # 计算所有病人的p_hat
        p_inf = w_predict/normalize_term # 计算无穷点的weight

        p_hat = np.append(p_hat,[p_inf])
        return p_hat
 
    # 计算对应的置信区间，输入nonconformal score, normalized weight p_hat, p_inf,以及指定的percentile
    def compute_quantile(self,t,p_hat,exp_g_x):
        exp_g_x_r = self.compute_nonconformal_score_single(t)
        if exp_g_x_r is None:
            return 1
        V_x = np.log(exp_g_x)-np.log(np.sum(exp_g_x_r))
        p_hat_leave = p_hat[self.V<=V_x[0]]
        return sum(p_hat_leave)
    
    def weighted_conformal_prediction(self,x,percentile=0.95):
        
        ch = self.model.predict_cumulative_hazards(x)
        exp_g_x = ch.loc[self.non_zero_idx]/self.bh

        p_hat = self.compute_normalized_weight(x)

        if percentile < 0.5:
            quantile = 1
            t = 5
            quantile = self.compute_quantile(t,p_hat,exp_g_x)
            while (quantile > percentile):
                step = t*(quantile-percentile)
                if step < 0.01:
                    step = 0.01
                t = t - sgn*step
                if int(t) < 0:
                    t = 0
                    break
                exp_g_x_r = self.compute_nonconformal_score_single(t)
                V_x = np.log(exp_g_x)-np.log(np.sum(exp_g_x_r))
                quantile = sum(p_hat[self.V<=V_x[0]])
                print(quantile,t)

            t_l = 0
            t_h = t
        else:
            quantile_l,quantile_h = 1,0
            t_l = 5
            t_h = 5
            quantile_l = self.compute_quantile(t_l,p_hat,exp_g_x)
            while (quantile_l>(1-percentile)/2):
                step_l = t_l*(quantile_l-(1-percentile)/2)
                if step_l < 0.01:
                    step_l = 0.01
                t_l = t_l - step_l
                if int(t_l) <= 0:
                    t_l = 0
                    break
                exp_g_x_r = self.compute_nonconformal_score_single(t_l)
                if exp_g_x_r is None:
                    quantile_l = 1
                    continue
                V_x = np.log(exp_g_x)-np.log(np.sum(exp_g_x_r))
                quantile_l = sum(p_hat[self.V<=V_x[0]])
                # print(quantile_l,t_l)
                
            quantile_h = self.compute_quantile(t_h,p_hat,exp_g_x)
            while (quantile_h<(0.5+self.percentile/2)):
                step_h = t_h*(0.5+self.percentile/2-quantile_h)
                if step_h < 0.01:
                    step_h = 0.01
                t_h = t_h + step_h
                exp_g_x_r = self.compute_nonconformal_score_single(t_h)
                if exp_g_x_r is None:
                    quantile_h = 1
                    continue
                V_x = np.log(exp_g_x)-np.log(np.sum(exp_g_x_r))
                # print(V_x)
                quantile_h = sum(p_hat[self.V<=V_x[0]])
            
        return (t_l, t_h)

    def run_training_step(self):
        print('--'*30)
        print('Begin Preprcessing Algorithm 1')
        self.run_preprocessing()
        print('--'*30)
        try:
            self.load_parameters(V_path = './model_data/V_alg_1.txt',W_path='./model_data/W_alg_1.txt',bh_path='./model_data/bh.txt',clf_path='./model_data/clf.model',model_path='./model_data/net_cox.model')
            print('Loading Parameters From Files')
        except:
            print('Begin Noncoformal Score')
            self.run_compute_nonconformal_score()
            print('--'*30)
            print('Begin Compute Wieght')
            self.run_conpute_weight()
            self.save_parmeters(V_path = './model_data/V_alg_1.txt',W_path='./model_data/W_alg_1.txt',bh_path='./model_data/bh.txt',clf_path='./model_data/clf.model',model_path='./model_data/net_cox.model')
        plt.hist(self.W,bins=100)
        plt.savefig('W.pdf')
        print(len(self.W),len(self.V))
            
    def get_T(self,x,percentile=0.95):
        t_l,t_h = self.weighted_conformal_prediction(x,percentile)
        return (t_l,t_h)
    
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

    def save_parmeters(self,V_path = 'V_alg_1.txt',W_path='W_alg_1.txt',bh_path='bh.txt',clf_path='clf.model',model_path='net_cox.model'):
        np.savetxt(V_path,self.V)
        np.savetxt(W_path,self.W)
        np.savetxt(bh_path,np.array([self.bh,self.non_zero_idx]))
        joblib.dump(self.clf,clf_path)
        joblib.dump(self.model,model_path)


    def load_parameters(self,V_path = 'V_alg_1.txt',W_path='W_alg_1.txt',bh_path='bh.txt',clf_path='clf.model',model_path='net_cox.model'):
        self.V = np.loadtxt(V_path)
        self.W = np.loadtxt(W_path)
        self.bh = float(np.loadtxt(bh_path)[0])
        self.non_zero_idx = float(np.loadtxt(bh_path)[1])
        self.clf = joblib.load(clf_path)
        self.model = joblib.load(model_path)
