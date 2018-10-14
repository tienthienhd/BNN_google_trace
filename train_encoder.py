import sys
import datetime
import shutil
import argparse
import os
import pandas as pd
import utils
from preprocessing_data import Data
from encoder_decoder import EncoderDecoder
import multiprocessing as mp


def run_config(config_):
    results_dir, config, idx = config_
    print('Start config', idx)
    dataset = Data(data_path='data/'+config['name'], use_features=config['features'])
    model = EncoderDecoder(unit_type=config['rnn_unit_type'],
                           activation=config['activation'],
                           layers_units=config['layers_units'],
                           input_keep_prob=config['input_keep_prob'],
                           output_keep_prob=config['output_keep_prob'],
                           state_keep_prob=config['state_keep_prob'],
                           variational_recurrent=config['variational_recurrent'],
                           optimizer=config['optimizer'],
                           batch_size=config['batch_size'],
                           num_features=len(config['features']),
                           learning_rate=config['learning_rate'])

    model.fit(dataset=dataset,
              num_epochs=config['num_epochs'],
              patience=config['patience'],
              encoder_sliding=config['sliding_encoder'],
              decoder_sliding=config['sliding_decoder'],
              log_name=results_dir+str(idx))
    model.close_sess()
    print('Finish config', idx)
    print('=========================================================')


def single_processing(results_dir, start_config):
    configs_ed = utils.read_config(results_dir + 'configs_ed.csv', 1000, start_config)

    for configs in configs_ed:
        for idx, config in configs.iterrows():
            run_config((results_dir, config, idx+start_config))


def multi_processing(results_dir, start_config):
    num_cpus = mp.cpu_count()
    pool = mp.Pool(num_cpus)

    configs_ed = utils.read_config(results_dir + 'configs_ed.csv', 1000, start_config)
    list_configs = []
    for configs in configs_ed:
        for idx, config in configs.iterrows():
            list_configs.append((results_dir, config, idx+start_config))
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


    results_dir = 'results/ed/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    if start_config == 0 and os.path.exists('results/ed/mae_rmse_log.csv'):
        os.remove('results/ed/mae_rmse_log.csv')
    if not os.path.exists(results_dir + 'configs_ed.csv') or new_config == 1:
        utils.generate_config(json_config_file, results_dir + 'configs_ed.csv',
                              ['data', 'encoder_decoder'])
    if multi_p == 1:
        multi_processing(results_dir, start_config)
    else:
        single_processing(results_dir, start_config)


if __name__ == '__main__':
    main()
