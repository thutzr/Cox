import numpy as np
import pandas as pd 
from pycox.simulations import SimStudyNonLinearNonPH

def generate_data(n):
      sim = SimStudyNonLinearNonPH()
      data = sim.simulate(n)
      df = sim.dict2df(data, True)
      df.drop(columns=['event_true','censoring_true'],inplace = True)
      df.to_pickle('./data/rr_nl_nph.pkl')

def load_data(path):
      df = pd.read_pickle(path)
      return df

if __name__ == '__main__':
      n = 10000
      generate_data(n)
      