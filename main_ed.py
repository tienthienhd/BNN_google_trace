# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 01:34:10 2018

@author: tienthien
"""

import multiprocessing as mp
import os
import pandas as pd
from encoder_decoder import EncoderDecoder
from data import Data
import sys
import datetime
import utils


def tuning_encoder_model(inputs):
    dataset = inputs[0]
    encoder_decoder_config = inputs[1]
    ed_dir = inputs[2]
    config_name = inputs[3]
    
    print('Start:', config_name)
#    print(dataset.get_max_min())
    ed_model = EncoderDecoder(encoder_decoder_config, dataset.get_max_min())
    
    # prepare data
    dataset.prepare_data_inputs_encoder(encoder_decoder_config['sliding_encoder'],
                                encoder_decoder_config['sliding_decoder'])
    train = dataset.get_data_encoder('train')
    val = dataset.get_data_encoder('val')
    test = dataset.get_data_encoder('test')
    
#     fit encoder decoder model
    ed_model.fit(train, val, test, ed_dir, config_name, verbose=0)
    print('directory:{}  {}'.format(ed_dir, config_name))



def single_processing(config_dir, start_config):
    # get config
    configs_read = utils.read_config(config_dir+'configs.csv', 1000, start_config)
    # loop config
    for configs in configs_read:
        for idx, config in configs.iterrows():
            dataset = Data(config)
            config_name = 'result_config_' + str(config[0])
            tuning_config = (dataset, config, config_dir, config_name)
            tuning_encoder_model(tuning_config)


def multi_processing(config_dir, pool, start_config):
#    list_configs = []
    configs_read = utils.read_config(config_dir+'configs.csv', 1000, start_config)
    # loop config
    for configs in configs_read:
        for idx, config in configs.iterrows():
            dataset = Data(config)
            config_name = 'result_config_' + str(config[0])
            tuning_config = (dataset, config, config_dir, config_name)
            pool.apply_async(tuning_encoder_model, args=(tuning_config, ))
#            list_configs.append(tuning_config)
                    
#            if len(list_configs) == 64:
#                pool.map(tuning_encoder_model, list_configs)
#                list_configs.clear()
                        
#    if len(list_configs) > 0:
#        pool.map(tuning_encoder_model, list_configs)
#        list_configs.clear()


def main():
    if len(sys.argv) < 3:
        print('Not enough parameters. Please put json filename and mode running')
    else:
        
        log_dir = './log/'
        results_dir = log_dir + 'results_' + str(datetime.datetime.now().date()) + '/'
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        
        json_config_file = sys.argv[1]
        mode_running = sys.argv[2]
        start_config = 0
        if len(sys.argv) == 4:
            start_config = int(sys.argv[3])
        
        if not os.path.exists(results_dir+'configs.csv'):
            utils.generate_config(json_config_file, results_dir)
        
        
        
        if mode_running == 'multi':
            num_processes = mp.cpu_count()
            print('Run on', num_processes, 'process')
            pool = mp.Pool(2)
            multi_processing(results_dir, pool, start_config)
            pool.close()
            pool.join()
            pool.terminate()
        else:
            single_processing(results_dir, start_config)
        
        
if __name__ == '__main__':
    main()