#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 19:24:36 2018

@author: tienthien
"""

import pandas as pd
import numpy as np

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n...t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1,...t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with Nan values
    if(dropnan):
        agg.dropna(inplace=True)
    return agg
    
def split_data(data, val_size=0.2, test_size=0.2):   
        ntest = int(round(len(data) * (1 - test_size)))
        nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))
     
        df_train = data.iloc[:nval]
        df_val = data.iloc[nval:ntest]
        df_test = data.iloc[ntest:]
        
        data = dict()
        data['train'] = df_train
        data['val'] = df_val
        data['test'] = df_test
        
        return data

class Data(object):
    def __init__(self, config):
        datapath = config["data_directory"] + config["data_name"] + '.csv'
        df = pd.read_csv(datapath, header=None, 
                         names = config['columns_full'])
        df = df.loc[:, config['features'][0]]
#        df.plot()
#        df = df.loc[:, config.common['features']]
        
        self.output_feature = config['features'][1][0]
        self.num_features = df.shape[1]
        self.df = df
        
        self.normalize()
        
        self.sliding = dict()
        
        
    def normalize(self):
        self.min = dict()
        self.max = dict()
        tmp = dict()
        for col in self.df.columns.values:
            data_col = self.df.loc[:, col].values
            self.min[col] = np.amin(data_col)
            self.max[col] = np.amax(data_col)
            tmp[col] = (data_col - self.min[col]) / (self.max[col] - self.min[col])
            
        self.df = pd.DataFrame(tmp)
        
    def get_max_min(self):
        return self.max[self.output_feature], self.min[self.output_feature]
    
    def denormalize(self, data, feature):
        min_ = self.min[feature]
        max_ = self.max[feature]
        
        tmp = data * (max_ - min_) + min_
        return tmp
    
    def prepare_data_inputs_encoder(self, encoder_sliding=1, decoder_sliding=1):
        self.sliding['encoder'] = encoder_sliding
        self.sliding['decoder'] = decoder_sliding
        
        data_encoder = series_to_supervised(self.df, encoder_sliding, 1)
        data_decoder = series_to_supervised(self.df, decoder_sliding, 1)
        
        data = pd.concat([data_encoder, data_decoder], axis=1)
        data.dropna(inplace=True)
        
        self.data = split_data(data)
        
    def prepare_data_inputs_mlp(self, input_dim=1):
        self.sliding['mlp'] = input_dim
        data = series_to_supervised(self.df, input_dim, 1)
        self.data = split_data(data)
        
        
    def get_data_encoder(self, dataset):
        data = self.data[dataset] # train, val or test
        data = data.astype(np.float32)
        
        index_encoder_x = 0
        index_encoder_y = self.sliding['encoder'] * self.num_features
        
        index_decoder_x = index_encoder_y + self.num_features
        index_decoder_y = index_decoder_x + self.sliding['decoder'] * self.num_features
        
        encoder_x = data.iloc[:, index_encoder_x:index_encoder_y].values
        encoder_x = encoder_x.reshape((encoder_x.shape[0], self.sliding['encoder'], 
                                       self.num_features))
        
        encoder_y = data.iloc[:, index_encoder_y:index_encoder_y+1].values
        
        
        decoder_x = data.iloc[:, index_decoder_x:index_decoder_y].values
        decoder_x = decoder_x.reshape((decoder_x.shape[0], self.sliding['decoder'], 
                                       self.num_features))
        
        decoder_y = data.iloc[:, index_decoder_y:index_decoder_y+1].values
        
        return encoder_x, decoder_x, decoder_y
    
    def get_data_mlp(self, dataset):
        data = self.data[dataset]
        data = data.astype(np.float32)
        
        index_x = 0
        index_y = self.sliding['mlp'] * self.num_features
        
        x = data.iloc[:, index_x:index_y].values
        x = x.reshape((x.shape[0], self.sliding['mlp'], self.num_features))
        
        y = data.iloc[:, index_y:index_y+1].values
        
        return x, y
        