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
from pycox.models import CoxPH
np.random.seed(1234)
_ = torch.manual_seed(123)
import warnings
warnings.filterwarnings('ignore')
import joblib
from pycox.simulations import SimStudyNonLinearNonPH
from generate_data import load_data
from scipy.spatial.distance import pdist, squareform
from weighted_conformal_prediction_coxph import WeightedConformalPrediction
from datetime import datetime 
import argparse
from pycox.datasets import support 
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
      method = 'cox_ph'
      data = 'support'
      print('--'*30+'|')
      print('[Method]\t'+method+'\t[Dataset]\t'+data+'\t[Alpha]\t'+str(alpha))
      df = support.read_df()
      df_data = df
      epochs = 10
      train_frac = 0.8
      empirical_coverage = []
      empirical_coverage_censor = []
      empirical_coverage_non_censor = []
      empirical_coverage_2 = []
      empirical_coverage_censor_2 = []
      empirical_coverage_non_censor_2 = []
      interval_length_1 = []
      interval_length_2 = []
      for epoch in range(epochs):
            rng = np.random.RandomState(epoch)
            shuffle_idx = rng.permutation(range(len(df_data)))
            train_idx = shuffle_idx[:int(train_frac*len(df_data))]
            calib_test_idx = shuffle_idx[int(train_frac*len(df_data)):]
            df_train = df_data.iloc[train_idx,:]
            df_calib_test = df_data.iloc[calib_test_idx,:]
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
            step = 0.1
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
                        else:
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
            n_calib_2_start_idx = X_test.shape[0] // 4

            calibration_indices = random_shuffle[:n_test_start_idx]
            calib_2_idx = random_shuffle[n_test_start_idx:n_test_start_idx+n_calib_2_start_idx]
            test_idx = random_shuffle[n_test_start_idx+n_calib_2_start_idx:]
            # 计算 calibration set 的 non conformal score
            print('--'*30+'|')
            print('Begin Computing Nonconformal Score of Alg. 1')
            df_calib = df_calib_test.iloc[calibration_indices,:]
            df_calib_2 = df_calib_test.iloc[calib_2_idx,:]
            df_test = df_calib_test.iloc[test_idx,:]

            
            df_calib_non_censor = df_calib[df_calib['event']==1]
            duration_test_non_censor,event_test_non_censor = wcp.get_target(df_calib_non_censor)
            calib_non_censor_idx = np.array(df_calib_non_censor.index)
            x_ca_non_censor = wcp.x_mapper.transform(df_calib_non_censor).astype('float32')
            cumulative_hazards_non_censor = wcp.model.predict_cumulative_hazards(x_ca_non_censor)
            exp_g_non_censor = cumulative_hazards_non_censor.loc[wcp.non_zero_idx]/wcp.bh

            try:
                  nonconformal_score_1 = np.loadtxt('./data/nonconformal_alg_1_'+str(epoch)+'_'+method+'_'+data+'_'+str(alpha)+'.txt')
            except:
                  nonconformal_score_1 = []
                  for i in range(len(df_calib_non_censor)):
                        t = int((duration_test_non_censor[i]-min_duration)/step)
                        exp_g_r = exp_R_list[t]
                        nonconformal_score_1.append(np.log(exp_g_non_censor[i])-np.log(exp_g_r))
                  
                  nonconformal_score_1.append(np.inf)
                  nonconformal_score_1 = np.array(nonconformal_score_1)
                  np.savetxt('./data/nonconformal_alg_1_'+str(epoch)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',nonconformal_score_1)

            sort_indices = np.argsort(nonconformal_score_1)
            sorted_nonconformal_score_1 = nonconformal_score_1[sort_indices]
            # 随机选取100个x_0
            print('--'*30+'|')
            print('Begin Testing')
            coverage_CI = []
            censor_coverage = []
            non_censor_coverage = []
            interval_len = []

            coverage_CI_2 = []
            censor_coverage_2 = []
            non_censor_coverage_2 = []
            interval_len_2 = []
            
            t1 = datetime.now()
            for test_point_idx in test_idx[:100]:
                  
                  weights = kernel_weights[test_point_idx][calib_non_censor_idx]
                  sampling_probs = kernel_weights[test_point_idx][test_idx] # 计算测试集中每个点的抽样概率
                  sampling_probs /= sampling_probs.sum()

                  # 对第二个calibration set中的点计算nonconformalty score
                  try:
                        nonconformal_score_2 = np.loadtxt('./data/nonconformal_alg_2_'+str(test_point_idx)+'_'+str(epoch)+'_'+method+'_'+data+'_'+str(alpha)+'.txt')
                        calibration_duration = np.loadtxt('./data/calibrated_duration_alg_2_'+str(test_point_idx)+'_'+str(epoch)+'_'+method+'_'+data+'_'+str(alpha)+'.txt')
                  except:
                        nonconformal_score_2 = []
                        calibrated_duration = []
                        for calib_point_2 in calib_2_idx:
                              sorted_calib_weights = np.append(weights[sort_indices[:-1]],kernel_weights[calib_point_2,test_point_idx])
                              sorted_calib_weights /= sorted_calib_weights.sum()
                              sorted_calib_weight_cum_sum = np.cumsum(sorted_calib_weights)
                              CI_calib_idx = np.min(np.argwhere(sorted_calib_weight_cum_sum>=alpha))
                              
                              threshold_score_calib = sorted_nonconformal_score_1[int(CI_calib_idx)]
                              
                              x_test = X_test[calib_point_2].reshape(1,-1)
                              cumulative_hazards_calib_2 = wcp.model.predict_cumulative_hazards(x_test)
                              exp_g_calib_2 = cumulative_hazards_calib_2.loc[wcp.non_zero_idx]/wcp.bh
                              exp_g_r_calib_2 = np.exp(np.log(exp_g_calib_2) - threshold_score_calib)
                              candidate_calib_idx = np.argwhere(exp_R_list>=exp_g_r_calib_2[0])

                              if len(candidate_calib_idx) == 0:
                                    t_calib = max_duration
                              else:
                                    t_calib = np.max(candidate_calib_idx)*step + min_duration
                              nonconformal_score_2.append(max(-duration_test[calib_point_2],duration_test[calib_point_2]-t_calib))

                              calibrated_duration.append(t_calib)
                        nonconformal_score_2.append(np.inf)
                        nonconformal_score_2 = np.array(nonconformal_score_2)
                        np.savetxt('./data/nonconformal_alg_2_'+str(test_point_idx)+'_'+str(epoch)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',nonconformal_score_2)

                        np.savetxt('./data/calibrated_duration_alg_2_'+str(test_point_idx)+'_'+str(epoch)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',np.array(calibrated_duration))
                  
                  sort_indices_2 = np.argsort(nonconformal_score_2)
                  sorted_nonconformal_score_2 = nonconformal_score_2[sort_indices_2]
                  
                  # 再选择100个作为test point，以对应的抽样概率抽样
                  included_1 = 0
                  included_censor_1 = 0
                  included_non_censor_1 = 0

                  included_2 = 0
                  included_censor_2 = 0
                  included_non_censor_2 = 0
                  
                  
                  sample_test_idx = rng.choice(test_idx,size=100,p=sampling_probs)
                  num_non_censor = sum(event_test[sample_test_idx])
                  interval_test_1 = []
                  interval_test_2 = []
                  for test_point_idx2 in sample_test_idx:
                        sorted_weights = np.append(weights[sort_indices[:-1]],kernel_weights[test_point_idx2,test_point_idx])
                        sorted_weights /= sorted_weights.sum()
                        sorted_weight_cum_sum = np.cumsum(sorted_weights)
                        CI_idx = np.min(np.argwhere(sorted_weight_cum_sum>=alpha))
                        
                        threshold_score = sorted_nonconformal_score_1[int(CI_idx)]
                        
                        x_test = X_test[test_point_idx2].reshape(1,-1)
                        cumulative_hazards_test = wcp.model.predict_cumulative_hazards(x_test)
                        exp_g_test = cumulative_hazards_test.loc[wcp.non_zero_idx]/wcp.bh
                        exp_g_r_test = np.exp(np.log(exp_g_test) - threshold_score)
                        candidate_idx = np.argwhere(exp_R_list<=exp_g_r_test[0])

                        if len(candidate_idx) == 0:
                              t_test_1 = max_duration
                        else:
                              t_test_1 = np.min(candidate_idx)*step + min_duration
                        interval_test_1.append(t_test_1)
                        if t_test_1 >= duration_test[test_point_idx2]:
                              included_1 += 1
                              if (event_test[test_point_idx2] == 0):
                                    included_censor_1 += 1
                              else:
                                    included_non_censor_1 += 1
                        
                        # 计算测试点在第二个算法下的区间
                        sorted_weights_2 = np.append(weights[sort_indices_2[:-1]],kernel_weights[test_point_idx2,test_point_idx])
                        sorted_weights_2 /= sorted_weights_2.sum()
                        sorted_weight_cum_sum_2 = np.cumsum(sorted_weights_2)
                        CI_idx_2 = np.min(np.argwhere(sorted_weight_cum_sum_2>=alpha))
                        
                        threshold_score_2 = sorted_nonconformal_score_2[int(CI_idx_2)]
                        if threshold_score_2 >= -t_test_1/2:
                              t_test_2 = t_test_1 + threshold_score_2
                        else:
                              t_test_2 = -threshold_score_2

                        interval_test_2.append(t_test_2)

                        if t_test_2 >= duration_test[test_point_idx2]:
                              included_2 += 1
                              if (event_test[test_point_idx2] == 0):
                                    included_censor_2 += 1
                              else:
                                    included_non_censor_2 += 1

                  if num_non_censor == 0:
                        non_censor_coverage.append(alpha)
                        non_censor_coverage_2.append(alpha)
                        censor_coverage.append(included_censor_1/(100-num_non_censor))
                        censor_coverage_2.append(included_censor_2/(100-num_non_censor))
                  elif num_non_censor == 100:
                        censor_coverage.append(alpha)
                        censor_coverage_2.append(alpha)
                        non_censor_coverage.append(included_non_censor_1/num_non_censor)
                        non_censor_coverage_2.append(included_non_censor_2/num_non_censor)
                  else:
                        censor_coverage.append(included_censor_1/(100-num_non_censor))
                        censor_coverage_2.append(included_censor_2/(100-num_non_censor))
                        non_censor_coverage.append(included_non_censor_1/num_non_censor)
                        non_censor_coverage_2.append(included_non_censor_2/num_non_censor)
                        
                  coverage_CI.append(included_1/100)
                  interval_len.append(np.mean(interval_test_1))
                  coverage_CI_2.append(included_2/100)
                  interval_test_2 = np.array(interval_test_2)
                  interval_test_2 = interval_test_2[interval_test_2!=np.inf]
                  interval_len_2.append(np.mean(interval_test_2))
            t2 = datetime.now()
            print('Time Elaspsed:\t %.2f s'% ((t2-t1).total_seconds()))
            print('[Epoch%d]\t [Mean]\t %.3f\t[Std.]\t%.3f'%(epoch,np.mean(coverage_CI),np.mean(coverage_CI_2)))
            np.savetxt('./data/coverage_alg_1_'+str(epoch)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',np.array(coverage_CI))
            empirical_coverage.append(np.mean(coverage_CI))
            empirical_coverage_censor.append(np.mean(censor_coverage))
            empirical_coverage_non_censor.append(np.mean(non_censor_coverage))

            np.savetxt('./data/coverage_alg_2_'+str(epoch)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',np.array(coverage_CI_2))
            empirical_coverage_2.append(np.mean(coverage_CI_2))
            empirical_coverage_censor_2.append(np.mean(censor_coverage_2))
            empirical_coverage_non_censor_2.append(np.mean(non_censor_coverage_2))
            

            interval_length_1.append(np.mean(interval_len))
            interval_length_2.append(np.mean(interval_len_2))

      print('Empirical Coverage of %d Epochs:\t [Alg.1]\t %.3f [Alg.2]\t %.3f'%(epochs,np.mean(empirical_coverage),np.mean(empirical_coverage_2)))
      print('Censor Data of %d Epochs:\t [Alg.1]\t %.3f [Alg.2]\t %.3f'%(epochs,np.mean(empirical_coverage_censor),np.mean(empirical_coverage_censor_2)))
      print('Non-Censor Data of %d Epochs:\t [Alg.1]\t %.3f [Alg.2]\t %.3f'%(epochs,np.mean(empirical_coverage_non_censor),np.mean(empirical_coverage_non_censor_2)))

      np.savetxt('./data/empirical_coverage_alg_1_'+str(epochs)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',empirical_coverage)
      np.savetxt('./data/empirical_coverage_censor_alg_1_'+str(epochs)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',empirical_coverage_censor)
      np.savetxt('./data/empirical_coverage_non_censor_alg_1_'+str(epochs)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',empirical_coverage_non_censor)
      
      np.savetxt('./data/empirical_coverage_alg_2_'+str(epochs)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',empirical_coverage_2)
      np.savetxt('./data/empirical_coverage_censor_alg_2_'+str(epochs)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',empirical_coverage_censor_2)
      np.savetxt('./data/empirical_coverage_non_censor_alg_2_'+str(epochs)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',empirical_coverage_non_censor_2)

      np.savetxt('./data/interval_length_alg_1_'+str(epochs)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',interval_length_1)
      np.savetxt('./data/interval_length_alg_2_'+str(epochs)+'_'+method+'_'+data+'_'+str(alpha)+'.txt',interval_length_2)

