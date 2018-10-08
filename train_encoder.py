import sys
import datetime
import shutil
import argparse
import os
import pandas as pd
import utils
from data import Data
from encoder_decoder import EncoderDecoder
import multiprocessing as mp


def run_config(config_):
    results_dir, config, idx = config_
    print('Start config', idx)
    dataset = Data(config)
    model = EncoderDecoder(config=config, 
                           max_min_data=dataset.get_max_min())
    model.fit(dataset, results_dir+str(idx))
    model.close_sess()
    print('Finish config', idx)
    print('=========================================================')


def single_processing(results_dir, start_config):
    configs_ed= utils.read_config(results_dir + 'configs_ed.csv', 1000, start_config)

    for configs in configs_ed:
        for idx, config in configs.iterrows():
            run_config((results_dir, config, idx))


def multi_processing(results_dir, start_config):
    num_cpus = mp.cpu_count()
    pool = mp.Pool(num_cpus)
    
    configs_ed= utils.read_config(results_dir + 'configs_ed.csv', 1000, start_config)
    list_configs = []
    for configs in configs_ed:
        for idx, config in configs.iterrows():
            list_configs.append((results_dir, config, idx))
            if len(list_configs) >= 64:
                pool.map(run_config, list_configs)
                list_configs.clear()
    if len(list_configs) > 0:
        pool.map(run_config, list_configs)
        list_configs.clear()
    
    pool.close()
    pool.join()
    pool.terminate()
    

def process_results_ed(results_dir): #FIXME: neu chi co cpu univariate
    if not os.path.exists('./log/models/'):
        os.mkdir('./log/models/')
    
    mae_rmse_file = results_dir + 'mae_rmse_log.csv'
    configs_file = results_dir + 'configs_ed.csv'
    
    df_mae_rmse = pd.read_csv(mae_rmse_file, names=['mae', 'rmse'])
    df_configs = pd.read_csv(configs_file, nrows=df_mae_rmse.shape[0])
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='Path of config file with format json')
    parser.add_argument('--start_config', help='Index start on list config', type=int)
    parser.add_argument('--new_config', help='Generate new config with config file', type=int, choices=[0,1])
    parser.add_argument('--mp', help='Running with multiprocess', type=int, choices=[0,1])
    args = parser.parse_args()
    
    json_config_file = args.config_file
    start_config = args.start_config
    if start_config is None:
        start_config = 0
    new_config = args.new_config
    multi_p = args.mp
    
        
#    print(json_config_file, start_config, new_config)
    
    log_dir = './log/'
    results_dir = log_dir + 'results_ed/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    if start_config == 0 and os.path.exists('./log/results_ed/mae_rmse_log.csv'):
        os.remove('./log/results_ed/mae_rmse_log.csv')
#    
    if not os.path.exists(results_dir + 'configs_ed.csv') or new_config == 1:
        utils.generate_config(json_config_file, results_dir+'configs_ed.csv', 
                              ['data', 'encoder_decoder'])
    if multi_p == 1:
        multi_processing(results_dir, start_config)
    else:
        single_processing(results_dir, start_config)
#    process_results_ed('./log/results_ed/')


if __name__ == '__main__':
    main()
    