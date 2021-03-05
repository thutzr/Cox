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
import warnings
warnings.filterwarnings('ignore')
import joblib
from pycox.simulations import SimStudyNonLinearNonPH
from generate_data import load_data
from scipy.spatial.distance import pdist, squareform
from weighted_conformal_prediction_coxph import WeightedConformalPrediction
from datetime import datetime
from pycox.datasets import metabric,support
alpha = 0.95

method = 'vanilla_cox_ph'
data = 'support'

df = support.read_df()
epochs = 10
train_frac = 0.6
test_frac = 0.2
val_frac = 0.2
coverage = []
coverage_censor = []
coverage_non_censor = []
for epoch in range(epochs):
      rng = np.random.RandomState(epoch)
      shuffle_idx = rng.permutation(range(len(df)))
      train_idx = shuffle_idx[:int(train_frac*len(df))]
      val_idx = shuffle_idx[int(train_frac*len(df)):int((train_frac+val_frac)*len(df))]
      test_idx = shuffle_idx[int((train_frac+val_frac)*len(df)):]

      df_train = df.iloc[train_idx,:]
      df_val = df.iloc[val_idx,:]
      df_test = df.iloc[test_idx,:]
      # cols_standardize = ['x0', 'x7', 'x8','x9','x10','x11','x12','x13']
      # cols_leave = ['x1', 'x2', 'x3', 'x4','x5','x6']
      # cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
      # cols_leave = ['x4', 'x5', 'x6', 'x7']
      cols_standardize = ['x0', 'x1', 'x2']
      # cols_leave = ['x1', 'x2', 'x3', 'x4','x5','x6']
      standardize = [([col], StandardScaler()) for col in cols_standardize]
      # polyfeature = [([col], PolynomialFeatures()) for col in cols_standardize]
      # leave = [(col, None) for col in cols_leave]
      x_mapper = DataFrameMapper(standardize)
      
      x_train = x_mapper.fit_transform(df_train).astype('float32')
      x_val = x_mapper.transform(df_val).astype('float32')
      x_test = x_mapper.transform(df_test).astype('float32')

      get_target = lambda df:(df['duration'].values,df['event'].values)
      y_train = get_target(df_train)
      y_val = get_target(df_val)
      duration_test, event_test = get_target(df_test)

      val = tt.tuplefy(x_val,y_val)

      in_features = x_train.shape[1]
      num_nodes = [32,32]
      out_features = 1
      batch_norm = True
      dropout = 0.1
      output_bias = False
      
      net = tt.practical.MLPVanilla(in_features=in_features, num_nodes=num_nodes, out_features = out_features, batch_norm = batch_norm, dropout = dropout, output_bias = output_bias)

      model = CoxPH(net,cd torch.optim.Adam)

      batch_size = 256
      n_epochs = 512
      verbose = False
      callbacks = [tt.callbacks.EarlyStopping()]
      model.fit(x_train,y_train,batch_size,n_epochs,callbacks,verbose,val_data=val.repeat(10).cat())
      _ = model.compute_baseline_hazards()

      surv = model.predict_surv_df(x_test)
      surv_ = (surv<=1-alpha).to_numpy(dtype='int8')
      index = np.array(surv.index)
      multiply_surv = np.transpose(surv_)*index
      multiply_surv_ = np.where(multiply_surv==0,np.max(index),multiply_surv)

      t_predict = multiply_surv_.min(axis = 1)
      diff_predict_true = np.subtract(t_predict,np.array(df_test['duration']))

      cover = sum(diff_predict_true>=0)/len(t_predict)
      censor = 0
      non_censor = 0
      for i in range(len(df_test)):
            if (diff_predict_true[i] >= 0):
                  if (event_test[i]==0):
                        censor += 1
                  else:
                        non_censor += 1
      
      coverage.append(cover)
      n_censor = len(df_test) - sum(df_test['event'])
      if n_censor == 0:
            coverage_censor.append(alpha)
            coverage_non_censor.append(cover)
      elif n_censor == len(df_test):
            coverage_non_censor.append(alpha)
            coverage_censor.append(cover)
      else:
            coverage_censor.append(censor/n_censor)
            coverage_non_censor.append(non_censor/(len(df_test)-n_censor))
            
      print('[%d]\t%.3f\t%.3f\t%.3f'%(epoch,cover,censor,non_censor))

print('Total Coverage Statistics:\t [Mean]%.3f\t[Std.]%.3f\t[Max]%.3f\t[Min]%.3f'%(np.mean(coverage),np.std(coverage),np.max(coverage),np.min(coverage)))

np.savetxt('/home/zeren/Research/Cox/output/anilla_coxph_coverage_'+data+str(epochs)+'.txt',np.array(coverage))
np.savetxt('/home/zeren/Research/Cox/output/vanilla_coxph_censor_coverage_'+data+str(epochs)+'.txt',np.array(coverage_censor))
np.savetxt('/home/zeren/Research/Cox/output/vanilla_coxph_non_censor_coverage_'+data+str(epochs)+'.txt',np.array(coverage_non_censor))