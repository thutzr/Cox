import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import pycox
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn_pandas import DataFrameMapper

# np.random.seed(1234)
# import warnings
# warnings.filterwarnings('ignore')
# import joblib
# from pycox.simulations import SimStudyNonLinearNonPH
from generate_data import load_data
import lifelines
from lifelines import CoxPHFitter

data = 'rr_nl_nph'
df = load_data('~/Research/Cox/data/rr_nl_nph.pkl')

time = [min(df['duration'])+i*0.01 for i in range(int((max(df['duration'])-min(df['duration']))/0.01))]

train_frac = 0.8
alpha = 0.95
epochs = 100
coverage = []
for epoch in range(epochs):
      rng = np.random.RandomState(epoch)
      shuffle_idx = rng.permutation(range(len(df)))
      train_idx = shuffle_idx[:int(train_frac*len(df))]
      test_idx = shuffle_idx[int(train_frac*len(df)):]
      df_train = df.iloc[train_idx,:]
      df_test = df.iloc[test_idx,:]
      df_train = df_train.drop(columns=['duration_true'])
      duration_true = df_test['duration_true']
      df_test = df_test.drop(columns=['duration_true'])
      cph = CoxPHFitter()
      cph.fit(df_train,duration_col = 'duration',event_col = 'event')

      surv = cph.predict_survival_function(df_test.iloc[:,:3],times=time)
      surv_ = (surv<=1-alpha).to_numpy(dtype='int8')
      index = np.array(surv.index)
      multiply_surv = np.transpose(surv_)*index
      multiply_surv_ = np.where(multiply_surv==0,np.max(index),multiply_surv)

      t_predict = multiply_surv_.min(axis = 1)
      diff_predict_true = np.subtract(t_predict,np.array(duration_true))

      cover = sum(diff_predict_true>=0)/len(t_predict)
      
      coverage.append(cover)
      print('[%d]\t%.3f'%(epoch,cover))

print('Total Coverage Statistics:\t [Mean]%.3f\t[Std.]%.3f\t[Max]%.3f\t[Min]%.3f'%(np.mean(coverage),np.std(coverage),np.max(coverage),np.min(coverage)))

np.savetxt('./data/cox_reg_coverage_'+str(epochs)+'.txt',np.array(coverage))