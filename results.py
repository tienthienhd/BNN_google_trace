# -*- coding: utf-8 -*-

from config import Config
from data import Data
from datetime import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

log_dir = './log/'
results_dir = log_dir + 'results/'


def read_config(filename):
    df_config = pd.read_csv(filename)
    return df_config

def read_mae_rmse(filename):
    df = pd.read_csv(filename, header=None)
    idx_mins = df.idxmin(axis=0)
    idx_min_mae = idx_mins[0]
    idx_min_rmse = idx_mins[1]
    return idx_mins, df.iloc[idx_min_mae, :].values, df.iloc[idx_min_rmse, :].values


def read_result_data(df_config_data):
    for idx, config in enumerate(df_config_data.values):
        print('data_config:' + str(idx) + ': ')
        print(config[2:4])
        data_config_dir = results_dir + 'data_config/' + str(idx) + '/'
        ed_result_dir = data_config_dir + 'encoder_decoder/'
        df_config_ed = read_config(data_config_dir + 'ed_config.csv')
        
        idx_mins, min_mae, min_rmse = read_mae_rmse(ed_result_dir+'mae_rmse_log.csv')
        
        
        idx_min_mae = idx_mins[0]
        idx_min_rmse = idx_mins[1]
        
        print('Min of mae at {}: mae={} rmse={}'.format(idx_min_mae, min_mae[0], min_mae[1]))
        print(df_config_ed.iloc[idx_min_mae, :])
        
#        img_history_min_mae = ed_result_dir + 'config_{}_history.png'.format(idx_min_mae)
#        img = plt.imread(img_history_min_mae)
#        plt.imshow(img)
#        plt.show()
#        plt.cla()
        
        print('=============================================================')


df_config_data = read_config(results_dir + 'data_config.csv')
read_result_data(df_config_data)
