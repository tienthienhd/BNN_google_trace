# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:56:49 2018

@author: tienthien
"""
import re 
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import ParameterGrid

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


#string =  ['[["meanCPUUsage", "canonical_memory_usage"], ["canonical_memory_usage"]]']
#a = parse_string_to_array_list(string)


    
def read_config(filename, chunksize=None, start_config=None):
    if chunksize:
        for chunk in pd.read_csv(filename, chunksize=chunksize, skiprows=range(1, start_config+1)):
            if 'layers_units' in chunk:
                chunk[['layers_units']] = chunk[['layers_units']].apply(parse_string_to_array_int)
            if 'columns_full' in chunk:
                chunk[['columns_full']] = chunk[['columns_full']].apply(parse_string_to_array_string)
            if 'features' in chunk:
                chunk[['features']] = chunk[['features']].apply(parse_string_to_array_list)
            yield chunk
    else:
        return pd.read_csv(filename)
            
      


def generate_config(json_file, config_dir):
    json_configs = None
    with open(json_file, 'r') as f:
        json_configs = json.load(f)
    
    data_ed_configs = json_configs['data']
    data_ed_configs.update(json_configs['encoder_decoder'])
    
    configs = list(ParameterGrid(data_ed_configs))
    df = pd.DataFrame(configs, index=None)
    df.to_csv(config_dir+'configs.csv', index=True)
    
    
        
#generate_config('test.json', './log/')
#a = read_config('./log/results_2018-09-16/configs.csv', 10000, 0)
#for i in a:
#    print(i)
              
#a = read_config_ed('./log/results_10_9/data_config/0/ed_config.csv', 1000)
#for i in a:
#    for y in i.columns:
#        print(y, i[y].dtype)
#    b = i['layers_units'].values
#    b = list(b)
#    break
#        
#a = read_config_data('./log/results_10_9/data_config.csv')
#for i in a:
#    for y in i.columns:
#        print(y, i[y].dtype)
#        
#    break
    

