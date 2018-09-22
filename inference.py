#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:11:32 2018

@author: tienthien
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data import Data
import utils
import math

def feed_forward(sess, encoder_x, prediction, X, batch_size):
    len_inputs = len(X)
    X = utils.padding(X, batch_size, pos=1)
    
    num_batches = int(len(X)/batch_size)
    if len(X) % batch_size != 0:
        num_batches += 1
    predictions = []
    for batch in range(num_batches):
        x = X[batch * batch_size : 
            (batch+1) * batch_size]
        pred = sess.run(prediction, feed_dict={encoder_x: x})
        pred = pred.reshape((batch_size))
        predictions.extend(pred)
        num_batches += 1
    return predictions[:len_inputs]
    
def mcdropout(sess, encoder_x, prediction, x, batch_size, num_iterator=10):
    y_preds = []
        
    for i in range(num_iterator):
        
        pred = feed_forward(sess, encoder_x, prediction, x, batch_size)
        y_preds.append(pred)
    
    y_preds = np.array(y_preds)
    y_mc = np.mean(y_preds, axis=0)
    
    theta = [[np.square(y-p) for p in y_preds[:, 0]] for y in y_mc]
    theta = np.array(theta)
    theta = np.mean(theta, axis=1)
    
#    biases = [np.square(y_mc - y_pred) for y_pred in y_preds]
#    theta = np.mean(biases)
    return y_mc, theta
    

def load_model(sess, saved_model):
    print('Load model:', saved_model)
    saver = tf.train.import_meta_graph(saved_model+'.meta')
    graph = tf.get_default_graph()
    
    encoder_x = graph.get_tensor_by_name('encoder_x:0')
    prediction = graph.get_tensor_by_name('prediction:0')
    
    saver.restore(sess, saved_model)
    return encoder_x, prediction

def inherent_noise(sess, encoder_x, prediction, val_set, batch_size): # FIXME
    X = val_set[0]
    Y = val_set[1]
    
    X = utils.padding(X, batch_size)
    Y = utils.padding(Y, batch_size)
    
    num_batches = int(len(X)/batch_size)
    if len(X) % batch_size != 0:
        num_batches += 1
    predictions = []
    
    for batch in range(num_batches):
        x = X[batch * batch_size : 
            (batch+1) * batch_size]
        pred = sess.run(prediction, feed_dict={encoder_x: x})
        pred = pred.reshape((batch_size))
        predictions.extend(pred)
        num_batches += 1
    theta = np.mean(np.square(predictions - Y[:len(predictions)]))
    return theta

def inference(sess, encoder_x, prediction, val_set, config, X):
    
    # mc dropout
    y_mc, theta_1 = mcdropout(sess, encoder_x, prediction, X, int(config['batch_size']), num_iterator=10)
    # inherent noise
    theta_2 = inherent_noise(sess, encoder_x, prediction, val_set, int(config['batch_size']))
    theta = theta_1 + theta_2
    theta = np.sqrt(theta_1 + theta_2)
    return y_mc, theta


def test():
    configs_read = utils.read_config('./log/models/configs_model_ed.csv', 1000, 0)
    configs = [a for a in configs_read]
    configs = configs[0]
    config = configs.iloc[3]
    
    data = Data(config)
    data.prepare_data_inputs_mlp(int(config['sliding_encoder']))
    val_set = data.get_data_mlp('val')
    test_set = data.get_data_mlp('test')
    
    tf.reset_default_graph()
    sess = tf.Session()
    # load model
    encoder_x, prediction = load_model(sess, './log/models/' + config['model_type'] + '_mlp.ckpt')
    
    y_pred, theta = inference(sess, encoder_x, prediction, val_set, config, test_set[0])
    sess.close()
    return y_pred, theta

y_pred, theta = test()