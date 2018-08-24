#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:48:26 2018

@author: tienthien
"""
#from multiprocessing import Pool
import multiprocessing as mp
import os
import pandas as pd
from encoder_decoder import EncoderDecoder
from config import Config
from data import Data
import sys

config_tuning_file = 'tuning_config.json'
if len(sys.argv) > 1:
    config_tuning_file = 'tuning_config (copy).json'

print(config_tuning_file)
exit()
log_dir = './log/results/'

def tuning_encoder_model(inputs):
    dataset = inputs[0]
    encoder_decoder_config = inputs[1]
    ed_dir = inputs[2]
    config_name = inputs[3]
    
    
#    print(dataset.get_max_min())
    ed_model = EncoderDecoder(encoder_decoder_config, dataset.get_max_min())
    
    # prepare data
    dataset.prepare_data_inputs_encoder(encoder_decoder_config['sliding_encoder'],
                                encoder_decoder_config['sliding_decoder'])
    train = dataset.get_data_encoder('train')
    val = dataset.get_data_encoder('val')
    test = dataset.get_data_encoder('test')
    
    print('directory:{}  {}'.format(ed_dir, config_name))
#    return config_name
    
#     fit encoder decoder model
    ed_model.fit(train, val, test, ed_dir, config_name, verbose=0)
        

def run(pool=None):
    if pool:
        multi_processing(pool)
    else:
        single_processing()


def multi_processing(pool):
    config = Config()
    data_configs = config.generate_config('data', config_tuning_file)
    df_data_configs = pd.DataFrame(data_configs, index=None)
    df_data_configs.to_csv(log_dir+'data_config.csv', index=False)
    del df_data_configs
    
    
    config_list = []
    for data_index, data_config in enumerate(data_configs):
        data_dir = log_dir + 'data_config/' + str(data_index) + '/'
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        dataset = Data(data_config)
        
        ed_configs = config.generate_config('encoder_decoder')
        df_ed_configs = pd.DataFrame(ed_configs, index=None)
        df_ed_configs.to_csv(data_dir+'ed_config.csv', index=False)
        del df_ed_configs
        
        for ed_index, ed_config in enumerate(ed_configs):
            ed_dir = data_dir + 'encoder_decoder/'
            if not os.path.exists(ed_dir):
                os.makedirs(ed_dir)
            config_name = 'config_' + str(ed_index)
            ed_config['num_features'] = len(data_config['features'][0])
            
            tuning_config = (dataset, ed_config, ed_dir, config_name)
            config_list.append(tuning_config)
            
            if len(config_list) == 64:
                pool.map(tuning_encoder_model, config_list)
                config_list.clear()
                
                
    if len(config_list) > 0:
        pool.map(tuning_encoder_model, config_list)
        config_list.clear()
        
        
def single_processing():
    config = Config()
    data_configs = config.generate_config('data', config_tuning_file)
    df_data_configs = pd.DataFrame(data_configs, index=None)
    df_data_configs.to_csv(log_dir+'data_config.csv', index=False)
    del df_data_configs
    
    for data_index, data_config in enumerate(data_configs):
        data_dir = log_dir + 'data_config/' + str(data_index) + '/'
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        dataset = Data(data_config)
        
        ed_configs = config.generate_config('encoder_decoder')
        df_ed_configs = pd.DataFrame(ed_configs, index=None)
        df_ed_configs.to_csv(data_dir+'ed_config.csv', index=False)
        del df_ed_configs
        
        for ed_index, ed_config in enumerate(ed_configs):
            ed_dir = data_dir + 'encoder_decoder/'
            if not os.path.exists(ed_dir):
                os.makedirs(ed_dir)
                
            config_name = 'config_' + str(ed_index)
            ed_config['num_features'] = len(data_config['features'][0])
            
            tuning_config = (dataset, ed_config, ed_dir, config_name)
            tuning_encoder_model(tuning_config)

def main():
    
    num_processes = mp.cpu_count()
    pool = mp.Pool(num_processes)
    
    
    run(pool)
    
    
    
    pool.close()
    pool.join()
    pool.terminate()           
    
            
            
main()