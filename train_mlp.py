import sys
import datetime
import shutil
import argparse
import os
import pandas as pd
import utils
from preprocessing_data import Data
from mlp import MLP
import multiprocessing as mp

def run_config(config_):
    results_dir, config, idx, encoder_model = config_
    print('Start config', idx)
    dataset = Data(data_path='data/' + config['name'], use_features=config['features'])
    model = MLP(encoder_model=encoder_model,
                activation=config['activation'],
                hidden_layers=config['hidden_layers'],
                optimizer=config['optimizer'],
                dropout_rate=config['dropout_rate'],
                learning_rate=config['learning_rate'],
                sliding=config['sliding_inference'])

    model.fit(dataset=dataset,
              num_epochs=config['num_epochs'],
              patience=config['patience'],
              sliding=config['sliding_inference'],
              log_name=results_dir+str(idx))
    model.close_sess()
    print('Finishing config', idx)
    print('===========================================')


def single_processing(results_dir, start_config, encoder_model):
    configs_ed = utils.read_config(results_dir + 'configs_mlp.csv', 1000, start_config)

    for configs in configs_ed:
        for idx, config in configs.iterrows():
            run_config((results_dir, config, idx, encoder_model))


def multi_processing(results_dir, start_config, encoder_model):
    num_cpus = mp.cpu_count()
    pool = mp.Pool(num_cpus)

    configs_mlp = utils.read_config(results_dir+'configs_mlp.csv', 1000, start_config)
    list_configs = []
    for configs in configs_mlp:
        for idx, config in configs.iterrows():
            list_configs.append((results_dir, config, idx, encoder_model))
            if len(list_configs) >= 64:
                pool.map(run_config, list_configs)
                list_configs.clear()
    if len(list_configs) > 0:
        pool.map(run_config, list_configs)
        list_configs.clear()

    pool.close()
    pool.join()
    pool.terminate()


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
    print(encoder_model)
    print("============================================")

    results_dir = 'results/mlp/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    if start_config == 0 and os.path.exists('results/mlp/mae_rmse_log.csv'):
        os.remove('results/mlp/mae_rmse_log.csv')

    if not os.path.exists(results_dir + 'configs_mlp.csv') or new_config == 1:
        utils.generate_config(json_config_file, results_dir + 'configs_mlp.csv',
                              ['data', 'mlp'])

    if multi_p == 1:
        multi_processing(results_dir, start_config, encoder_model)
        pass
    else:
        single_processing(results_dir, start_config, encoder_model)


if __name__ == '__main__':
    main()
