#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 19:24:16 2018

@author: tienthien
"""

import re 
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import tensorflow as tf


def parse_string_to_array_int(string):
    results = []
    for s in string:
        objects = re.findall('\d+', s)
        objects = list(map(int, objects))
        results.append(objects)
    return results


def parse_string_to_array_string(string):
    results = []
    for s in string:
        objects = re.findall('\w+', s)
        objects = list(map(str, objects))
        results.append(objects)
    return results


def parse_string_to_array_list(string):
    results = []
    for s in string:
        i_start = s.index(']') + 1
        objects = [s[1:i_start], s[i_start+1: s.index(']', i_start)+1]]
        objects = parse_string_to_array_string(objects)
        results.append(objects)
    return results

    
def read_config(filename, chunksize=None, start_config=0, num_configs=None):
    if chunksize is not None:
        for chunk in pd.read_csv(filename, chunksize=chunksize, skiprows=range(1, start_config+1), nrows=num_configs):
            if 'layers_units' in chunk:
                chunk[['layers_units']] = chunk[['layers_units']].apply(parse_string_to_array_int)
            if 'columns_full' in chunk:
                chunk[['columns_full']] = chunk[['columns_full']].apply(parse_string_to_array_string)
            if 'features' in chunk:
                chunk[['features']] = chunk[['features']].apply(parse_string_to_array_list)
            if 'hidden_layers' in chunk:
                chunk[['hidden_layers']] = chunk[['hidden_layers']].apply(parse_string_to_array_int)
            yield chunk
    else:
        return pd.read_csv(filename)
            

def generate_config(json_file, file_to_save, list_field=None):
    json_configs = None
    with open(json_file, 'r') as f:
        json_configs = json.load(f)
    
    configs_dict = dict()
    if list_field is None:
        for key, value in json_configs.items():
            configs_dict.update(value)
    else:
        for key in list_field:
            configs_dict.update(json_configs[key])
    
    configs = list(ParameterGrid(configs_dict))
    df = pd.DataFrame(configs, index=None)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(file_to_save, index=True)
    

def plot_log(log_dict, metrics=None, file_save=None):
    for key, value in log_dict.items():
        plt.plot(range(len(value)), value, label=key)
    if metrics:
        plt.xlabel(metrics[0])
        plt.ylabel(metrics[1])
    
    plt.legend()
    plt.savefig(file_save)
#    plt.show()
    plt.clf()


def padding(array, batch_size, pos=0, values=0):
    # padding_len = 0
    if batch_size < len(array):
        padding_len = batch_size - len(array) % batch_size
    else:
        return array[:batch_size]
    rank = len(array.shape)
    rank_pad_value = rank - 1
    shape_pad_value = []
    padding_values = []
    for i in range(rank_pad_value):
        shape_pad_value.append(array.shape[i+1])
    
    for i in range(padding_len):
        padding_values.append(np.full(shape_pad_value, values, dtype=np.float32))
    
    result = np.array(padding_values)
    if pos == 0:  # pad on head
        result = np.concatenate((result, array))
    elif pos == 1:  # pad on tail
        result = np.concatenate((array, result))
    
    return result


def early_stop(array, idx, patience=0, min_delta=0.0):
    if idx <= patience :
        return False
    
    value = array[idx - patience - 1]
    arr = array[idx-patience:]
    check = 0
    for val in arr:
        if(val - value > min_delta):
            check += 1
    if(check == patience):
        return True
    else:
        return False


def activation(name):
    if name == 'tanh':
        return tf.nn.tanh
    elif name == 'sigmoid':
        return tf.nn.sigmoid
    elif name == 'relu':
        return tf.nn.relu
    elif name == 'elu':
        return tf.nn.elu
    else:
        raise Exception(name + " is not available!")


def optimizer(name, learning_rate=0.001):
    if name == 'adam':
        return tf.train.AdamOptimizer()
    elif name == 'gradient':
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif name == 'momemtum':
        return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.7)
    elif name == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    else:
        raise Exception(name + " is not available!")

