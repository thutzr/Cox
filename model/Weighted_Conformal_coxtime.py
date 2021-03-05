import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pycox
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn_pandas import DataFrameMapper
import torch
import torchtuples as tt
from pycox.datasets import metabric
from pycox.evaluation import EvalSurv
from pycox.models import CoxTime
np.random.seed(1234)
_ = torch.manual_seed(123)
import warnings
warnings.filterwarnings('ignore')
import joblib
from pycox.simulations import SimStudyNonLinearNonPH
from generate_data import load_data
from scipy.spatial.distance import pdist, squareform
from weighted_conformal_prediction_coxtime import WeightedConformalPrediction
from datetime import datetime 
import argparse
import multiprocess
from multiprocess import Pool
# parser = argparse.ArgumentParser(description='Specify Alpha')
# parser.add_argument('-alpha',type=float,help='the confidence level')
# alpha = parser.parse_args()
alphas = [0.95]
# cores = multiprocess.cpu_count()
# pool = Pool(processes = cores-1)
# results = pool.map(Algorithm_1,alphas)
for alpha in alphas:
# def Algorithm_1(alpha):
      method = 'cox_time'
      data = 'rr_nl_nph'
      print('--'*30+'|')
      print('[Method]\t'+method+'\t[Dataset]\t'+data+'\t[Alpha]\t'+str(alpha))
      df = load_data('~/Cox/data/rr_nl_nph.pkl')
      df_data = df
      epochs = 100
      train_frac = 0.8
      empirical_coverage = []
      empirical_coverage_censor = []
      empirical_coverage_non_censor = []
      for epoch in range(epochs):
            rng = np.random.RandomState(epoch)
            shuffle_idx = rng.permutation(range(len(df_data)))
            train_idx = int(train_frac*len(df_data))
            df_train = df_data.iloc[:train_idx,:]
            df_calib_test = df_data.iloc[train_idx:,:]
            df_calib_test.reset_index(drop = True,inplace = True)
            print('--'*30+'|')
            print('Initializing Algorithms')
            wcp = WeightedConformalPrediction(df_train,verbose = False,percentile=alpha)
            wcp.run_preprocessing()
            wcp.find_baseline_hazard_non_zero_idx()
            # 对所有可能的时间计算对用的at risk的人，以及相应的exp_g
            print('--'*30+'|')
            print('Pre-compute Hazard At Risk')
            max_duration = np.max(df_data['duration'])
            min_duration = np.min(df_data['duration'])
            step = 0.01
            num_steps = int((max_duration-min_duration)/step)+1
            try:
                  exp_R_list = np.loadtxt('./data/exp_R_'+str(epoch)+'_'+method+'_'+data+'_'+str(alpha)+'.txt')
            except:
                  exp_R_list = []
                  for i in range(num_steps):
                        t = min_duration + i*step
                        at_risk = df_train[df_train['duration']>=t]
                        if len(at_risk) == 0:
                              exp_R_list.append(0)
                        x_risk = wcp.x_mapper.transform(at_risk).astype('float32')
                        cumulative_hazards_risk = wcp.model.predict_cumulative_hazards(x_risk)
                        exp_R_list.append(np.sum(cumulative_hazards_risk.loc[wcp.non_zero_idx]/wcp.bh))

                  np.savetxt('./data/exp_R_'+str(epoch)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',np.array(exp_R_list))
            X_test = wcp.x_mapper.transform(df_calib_test)
            duration_test,event_test = wcp.get_target(df_calib_test)
            
            sq_dists = squareform(pdist(X_test,'sqeuclidean'))
            kernel_weights = np.exp(-sq_dists)
            
            random_shuffle =rng.permutation(X_test.shape[0])
            n_test_start_idx = X_test.shape[0] // 2

            calibration_indices = random_shuffle[:n_test_start_idx]
            test_indices = random_shuffle[n_test_start_idx:]
            # 计算 calibration set 的 non conformal score
            print('--'*30+'|')
            print('Begin Computing Nonconformal Score')
            df_calib = df_calib_test.iloc[calibration_indices,:]
            df_test = df_calib_test.iloc[test_indices,:]
            df_calib_censor = df_calib[df_calib['event']==1]
            calib_censor_idx = np.array(df_calib_censor.index)
            x_ca_censor = wcp.x_mapper.transform(df_calib_censor).astype('float32')
            cumulative_hazards_censor = wcp.model.predict_cumulative_hazards(x_ca_censor)
            exp_g_censor = cumulative_hazards_censor.loc[wcp.non_zero_idx]/wcp.bh
            try:
                  nonconformal_score = np.loadtxt('./data/nonconformal_'+str(epoch)+'_'+method+'_'+data+'_'+str(alpha)+'.txt')
            except:
                  nonconformal_score = []
                  for i in range(len(df_calib_censor)):
                        t = int((duration_test[i]-min_duration)/step)
                        exp_g_r = exp_R_list[t]
                        nonconformal_score.append(np.log(exp_g_censor[i])-np.log(exp_g_r))
                  
                  nonconformal_score.append(np.inf)
                  nonconformal_score = np.array(nonconformal_score)
                  np.savetxt('./data/nonconformal_'+str(epoch)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',nonconformal_score)
            sort_indices = np.argsort(nonconformal_score)
            sorted_nonconformal_score = nonconformal_score[sort_indices]
            # 随机选取100个x_0
            print('--'*30+'|')
            print('Begin Testing')
            coverage_CI = []
            censor_coverage = []
            non_censor_coverage = []
            interval_len = []
            t1 = datetime.now()
            for test_point_idx in test_indices[:100]:
                  
                  weights = kernel_weights[test_point_idx][calib_censor_idx]
                  sampling_probs = kernel_weights[test_point_idx][test_indices] # 计算测试集中每个点的抽样概率
                  sampling_probs /= sampling_probs.sum()
                  # 再选择100个作为test point，以对应的抽样概率抽样
                  included = 0
                  included_censor = 0
                  included_non_censor = 0
                  
                  sample_test_idx = rng.choice(test_indices,size=100,p=sampling_probs)
                  num_non_censor = sum(event_test[sample_test_idx])
                  interval_test = []
                  for test_point_idx2 in sample_test_idx:
                        sorted_weights = np.append(weights[sort_indices[:-1]],kernel_weights[test_point_idx2,test_point_idx])
                        sorted_weights /= sorted_weights.sum()
                        sorted_weight_cum_sum = np.cumsum(sorted_weights)
                        CI_idx = np.min(np.argwhere(sorted_weight_cum_sum>=0.95))
                        
                        threshold_score = sorted_nonconformal_score[int(CI_idx)]
                        
                        x_test = X_test[test_point_idx2].reshape(1,-1)
                        cumulative_hazards_test = wcp.model.predict_cumulative_hazards(x_test)
                        exp_g_test = cumulative_hazards_test.loc[wcp.non_zero_idx]/wcp.bh
                        exp_g_r_test = np.exp(np.log(exp_g_test) - threshold_score)
                        candidate_idx = np.argwhere(exp_R_list<=exp_g_r_test[0])
                        if len(candidate_idx) == 0:
                              t_test = max_duration
                        else:
                              t_test = np.min(candidate_idx)*step + min_duration
                        interval_test.append(t_test)
                        if t_test >= duration_test[test_point_idx2]:
                              included += 1
                              if (event_test[test_point_idx2] == 0):
                                    included_censor += 1
                              else:
                                    included_non_censor += 1
                  
                  censor_coverage.append(included_censor/(100-num_non_censor))
                  non_censor_coverage.append(included_non_censor/num_non_censor)
                  coverage_CI.append(included/100)
                  interval_len.append(np.mean(interval_test))
            t2 = datetime.now()
            print('Time Elaspsed:\t %.2f s'% ((t2-t1).total_seconds()))
            print('[Epoch%d]\t [Mean]\t %.3f\t[Std.]\t%.3f'%(epoch,np.mean(coverage_CI),np.std(coverage_CI)))
            np.savetxt('./data/coverage_'+str(epoch)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',np.array(coverage_CI))
            empirical_coverage.append(np.mean(coverage_CI))
            empirical_coverage_censor.append(np.mean(censor_coverage))
            empirical_coverage_non_censor.append(np.mean(non_censor_coverage))
      print('Empirical Coverage of %d Epochs:\t [Mean]\t %.3f [Std.]\t %.3f'%(epochs,np.mean(empirical_coverage),np.std(empirical_coverage)))
      print('Censor Data of %d Epochs:\t\t [Mean]\t %.3f [Std.]\t %.3f'%(epochs,np.mean(empirical_coverage_censor),np.std(empirical_coverage_censor)))
      print('Non-Censor Data of %d Epochs:\t\t [Mean]\t %.3f [Std.]\t %.3f'%(epochs,np.mean(empirical_coverage_non_censor),np.std(empirical_coverage_non_censor)))
      np.savetxt('./data/empirical_coverage_'+str(epochs)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',empirical_coverage)
      np.savetxt('./data/empirical_coverage_censor_'+str(epochs)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',empirical_coverage_censor)
      np.savetxt('./data/empirical_coverage_non_censor_'+str(epochs)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',empirical_coverage_non_censor)
      # return 0


