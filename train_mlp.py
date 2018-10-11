import sys
import datetime
import pandas as pd
import shutil
import argparse
import os
import utils
from data import Data
from mlp import MLP


def run_config(config_, encoder_model):
    results_dir, config, idx = config_
    print('Start config', idx)
    dataset = Data(config)
    model = MLP(config=config, max_min_data=dataset.get_max_min(), model_encoder=encoder_model)
    model.fit(dataset, results_dir + str(idx))
    model.close_sess()
    print('Finish config', idx)
    print('=========================================================')


def get_type_model(config):
    features = config['features']
    input_features =features[0]
    output_feature = features[1][0]
    
    types = ['univariate', 'multivariate']
    if 'CPU' in output_feature:
        return 'cpu_' + types[len(input_features)-1] + '_ed'
    elif 'mem' in output_feature:
        return 'mem_' + types[len(input_features)-1] + '_ed'
    
#    return config['features']


def single_processing(results_dir, start_config, encoder_model):
    configs_ed = utils.read_config(results_dir + 'configs_mlp.csv', 1000, start_config)
    # df_ed_configs = pd.read_csv('./log/models/configs_model_ed.csv')
    
    for configs in configs_ed:
        for idx, config in configs.iterrows():
            run_config((results_dir, config, idx), encoder_model)


def process_results_mlp(results_dir):
    print('Processing results')
    if not os.path.exists('./log/models/'):
        os.mkdir('./log/models/')

    mae_rmse_file = results_dir + 'mae_rmse_log.csv'
    configs_file = results_dir + 'configs_mlp.csv'
    
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
    df_configs_min.to_csv('./log/models/configs_model_mlp.csv', index=False)
    
    # move saved model to special folder
    tail_files = ['_mlp.ckpt.data-00000-of-00001', '_mlp.ckpt.index', '_mlp.ckpt.meta']
    src_dir = './log/results_mlp/'
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
    parser.add_argument('--encoder_model', required=True, help='Path of encoder model')
    parser.add_argument('--start_config', help='Index start on list config', type=int)
    parser.add_argument('--new_config', help='Generate new config with config file', type=int, choices=[0, 1])
    parser.add_argument('--mp', help='Running with multiprocess', type=int, choices=[0, 1])
    args = parser.parse_args()
    
    json_config_file = args.config_file
    start_config = args.start_config
    if start_config is None:
        start_config = 0
    new_config = args.new_config
    multi_p = args.mp
    encoder_model = args.encoder_model
    
    log_dir = './log/'
    results_dir = log_dir + 'results_mlp/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    if start_config == 0 and os.path.exists('./log/results_mlp/mae_rmse_log.csv'):
        os.remove('./log/results_mlp/mae_rmse_log.csv')
    
    if not os.path.exists(results_dir + 'configs_mlp.csv') or new_config == 1:
        utils.generate_config(json_config_file, results_dir+'configs_mlp.csv', 
                              ['data', 'mlp'])

    if multi_p == 1:
        # multi_processing(results_dir, start_config)
        pass
    else:
        single_processing(results_dir, start_config, encoder_model)


if __name__ == '__main__':
    main()
    