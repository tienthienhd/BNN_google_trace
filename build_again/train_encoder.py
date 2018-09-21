#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 21:26:42 2018

@author: tienthien
"""

import sys
import datetime
import shutil
import os
import pandas as pd
import utils
from data import Data
from encoder_decoder import EncoderDecoder


def single_processing(results_dir, start_config):
    configs_ed= utils.read_config(results_dir + 'configs_ed.csv', 1000, start_config)

    for configs in configs_ed:
        for idx, config in configs.iterrows():
            print('Start config', idx)
            dataset = Data(config)
            model = EncoderDecoder(config=config, 
                                   max_min_data=dataset.get_max_min())
            model.fit(dataset, results_dir+str(idx))
            model.close_sess()
            print('Finish config', idx)
            print('=========================================================')

def process_results_ed(results_dir):
    mae_rmse_file = results_dir + 'mae_rmse_log.csv'
    configs_file = results_dir + 'configs_ed.csv'
    
    df_mae_rmse = pd.read_csv(mae_rmse_file, names=['mae', 'rmse'])
    df_configs = pd.read_csv(configs_file)
    
    df_resutls = pd.concat([df_configs, df_mae_rmse], axis=1)
    
    df_cpu_univariate = pd.DataFrame(columns=df_resutls.columns)
    df_cpu_multivariate = pd.DataFrame(columns=df_resutls.columns)
    df_mem_univariate = pd.DataFrame(columns=df_resutls.columns)
    df_mem_multivariate = pd.DataFrame(columns=df_resutls.columns)
    
    for idx, row in df_resutls.iterrows():
        if row['features'] == "[['meanCPUUsage'], ['meanCPUUsage']]":
            df_cpu_univariate = df_cpu_univariate.append(row, ignore_index=False)
        elif row['features'] == "[['meanCPUUsage', 'canonical_memory_usage'], ['meanCPUUsage']]":
            df_cpu_multivariate = df_cpu_multivariate.append(row, ignore_index=False)
        elif row['features'] == "[['canonical_memory_usage'], ['canonical_memory_usage']]":
            df_mem_univariate = df_mem_univariate.append(row, ignore_index=False)
        elif row['features'] == "[['meanCPUUsage', 'canonical_memory_usage'], ['canonical_memory_usage']]":
            df_mem_multivariate = df_mem_multivariate.append(row, ignore_index=False)
            
    idx_cpu_univariate_min = df_cpu_univariate['rmse'].idxmin()
    idx_cpu_multivariate_min = df_cpu_multivariate['rmse'].idxmin()
    idx_mem_univariate_min = df_mem_univariate['rmse'].idxmin()
    idx_mem_multivariate_min = df_mem_multivariate['rmse'].idxmin()
    
    # save configs have min error
    # save config get min rmse
    df_configs_min = df_resutls.iloc[[idx_cpu_univariate_min, 
                                     idx_cpu_multivariate_min, 
                                     idx_mem_univariate_min, 
                                     idx_mem_multivariate_min]].reset_index(drop=True)
    df_name = pd.DataFrame({'model_type':['cpu_univariate', 
                                          'cpu_multivariate', 
                                          'mem_univariate', 
                                          'mem_multivariate']})
    df_configs_min = pd.concat([df_name, df_configs_min.iloc[:, 1:]], axis=1)
    df_configs_min.to_csv('./log/models/configs_model_ed.csv', index=False)
    
    
    # move saved model to special folder
    tail_files = ['_ed.ckpt.data-00000-of-00001', '_ed.ckpt.index', '_ed.ckpt.meta']
    src_dir = './log/results_ed/'
    dest_dir = './log/models/'
    
    shutil.copy(src_dir + str(idx_cpu_univariate_min)+'_model' + tail_files[0], 
                dest_dir+'cpu_univariate'+tail_files[0])
    shutil.copy(src_dir + str(idx_cpu_univariate_min)+'_model' + tail_files[1], 
                dest_dir+'cpu_univariate'+tail_files[1])
    shutil.copy(src_dir + str(idx_cpu_univariate_min)+'_model' + tail_files[2], 
                dest_dir+'cpu_univariate'+tail_files[2])
    
    shutil.copy(src_dir + str(idx_cpu_multivariate_min)+'_model' + tail_files[0], 
                dest_dir+'cpu_multivariate'+tail_files[0])
    shutil.copy(src_dir + str(idx_cpu_multivariate_min)+'_model' + tail_files[1], 
                dest_dir+'cpu_multivariate'+tail_files[1])
    shutil.copy(src_dir + str(idx_cpu_multivariate_min)+'_model' + tail_files[2], 
                dest_dir+'cpu_multivariate'+tail_files[2])
    
    shutil.copy(src_dir + str(idx_mem_univariate_min)+'_model' + tail_files[0], 
                dest_dir+'mem_univariate'+tail_files[0])
    shutil.copy(src_dir + str(idx_mem_univariate_min)+'_model' + tail_files[1], 
                dest_dir+'mem_univariate'+tail_files[1])
    shutil.copy(src_dir + str(idx_mem_univariate_min)+'_model' + tail_files[2], 
                dest_dir+'mem_univariate'+tail_files[2])
    
    shutil.copy(src_dir + str(idx_mem_multivariate_min)+'_model' + tail_files[0], 
                dest_dir+'mem_multivariate'+tail_files[0])
    shutil.copy(src_dir + str(idx_mem_multivariate_min)+'_model' + tail_files[1], 
                dest_dir+'mem_multivariate'+tail_files[1])
    shutil.copy(src_dir + str(idx_mem_multivariate_min)+'_model' + tail_files[2], 
                dest_dir+'mem_multivariate'+tail_files[2])
    


def main():
    if len(sys.argv) < 2:
        print('Not enough parameters.')
        return None
    log_dir = './log/'
#    results_dir = log_dir + 'results_' + str(datetime.datetime.now().date()) + '/'
    results_dir = log_dir + 'results_ed/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    json_config_file = sys.argv[1]
    start_config = 0
    if len(sys.argv) == 3:
        start_config = int(sys.argv[2])
    
    if not os.path.exists(results_dir + 'configs_ed.csv') or sys.argv[3] == 'new_config':
        utils.generate_config(json_config_file, results_dir+'configs_ed.csv', 
                              ['data', 'encoder_decoder'])
    
    single_processing(results_dir, start_config)

if __name__ == '__main__':
    main()
    process_results_ed('./log/results_ed/')