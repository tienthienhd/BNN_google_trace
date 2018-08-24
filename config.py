# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 22:31:09 2018

@author: HP Zbook 15
"""

import json
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import ParameterGrid

def combination(array):
    b = []
    if len(array) <= 1:
        b = array
    else:
        b = list(itertools.product(array[0], array[1]))
        for i in range(2, len(array)):
            b = list(itertools.product(b, array[i]))
            for j in range(len(b)):
                c = list(b[j][0])
                c.append(b[j][1])
                b[j] = c
    return b

def generate(config_dict):
    columns = config_dict.keys()
    tmp = [config_dict[key] for key in columns]
    body = combination(tmp)
    data = pd.DataFrame(body)
    data.columns=columns
    return data
    
class Config(object):
    def __init__(self, filename=None):
        self.config = None
        if filename:
            with open(filename, 'r') as f:
                config = json.load(f)
                self.encoder_decoder = config['encoder_decoder']
                self.mlp = config['mlp']
                self.data = config['data']
            

#    def get_config_tuning(self, filename):
#        with open(filename, 'r') as f:
#            self.config_tuning = json.load(f)
#        return self.config_tuning
             
            
            
#    def generate_config(self, type_config):
#        config = self.config_tuning[type_config]
#        return generate(config)
    
    def load_config(self, filename):
        print('Loading config from:', filename)
        with open(filename, 'r') as f:
            self.config = json.load(f)
            
        print('Loaded config!')
            
            
    def generate_config(self, type_config, filename=None):
        if self.config is None:
            self.load_config(filename)
            
        config = self.config[type_config]
        
        configs = ParameterGrid(config)
        return list(configs)
    
        
#c = Config()
#a = c.generate_config('encoder_decoder')
