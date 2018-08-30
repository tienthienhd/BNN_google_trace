#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 17:36:41 2018

@author: tienthien
"""

import tensorflow as tf
import numpy as np


#import matplotlib
#matplotlib.use('Agg')


import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(train_losses, val_losses=None, file_save=None):
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, label='train loss')
    if val_losses:
        plt.plot(epochs, val_losses, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(file_save)
#    plt.show()
    plt.clf()
    
    
def early_stop(array, patience=0, min_delta=0.0):
        if len(array) <= patience :
            return False
        
        value = array[len(array) - patience - 1]
        arr = array[len(array)-patience:]
        check = 0
        for val in arr:
            if(val - value > min_delta):
                check += 1
        if(check == patience):
            return True
        else:
            return False
    
    
class MLP(object):
    def __init__(self, config=None, max_min=None):
        tf.reset_default_graph()
        
        elf.hidden_layers = config['hidden_layers']
        self.activation = tf.nn.tanh
        self.num_epochs = config['num_epochs']
        
        self.batch_size = config['batch_size']
        self.optimizer = tf.train.AdamOptimizer()
        self.dropout_rate = 0.9
        
        if max_min:
            self.max = max_min[0]
            self.min = max_min[1]
            
    def load_encoder(self, sess):
        encoder_saver = tf.train.import_meta_graph('./log/model/encoder_decoder.ckpt.meta')
        
        encoder_graph = tf.get_default_graph()
        
        self.x = encoder_graph.get_tensor_by_name('encoder_x:0')
        
        encoder_outputs = encoder_graph.get_tensor_by_name('encoder/encoder_outputs:0')
        
        output_encoder_sg = tf.stop_gradient(encoder_outputs)
    
        self.encoder_last_outputs = output_encoder_sg[:, :, -1]
        
#        self.encoder_state = encoder_graph.get_tensor_by_name('init_state:0')
        
        
        encoder_saver.restore(sess, './log/model/encoder_decoder.ckpt')
        
        
    def build_model(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None], name='x')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')
        
        prev_layer = layer = None
        for i, num_units in enumerate(self.hidden_layers):
            if i==0:
                layer = tf.layers.dense(inputs=self.encoder_last_outputs, 
                                        activation=self.activation, 
                                        units=num_units, 
                                        name='layer'+str(i))
            else:
                layer = tf.layers.dense(inputs=prev_layer, 
                                        activation=self.activation, 
                                        units=num_units, 
                                        name='layer'+str(i))
            prev_layer = layer
            layer = tf.layers.dropout(prev_layer, rate=self.dropout_rate)
            prev_layer = layer
        pred = tf.layers.dense(inputs=prev_layer, 
                                    units=1, 
                                    name='output_layer')
        
        self.pred_inverse = pred * (self.max + self.min) + self.min
        self.y_inverse = self.y * (self.max + self.min) + self.min
        
        self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.pred_inverse, 
                                                     self.y_inverse)))
        self.RMSE = tf.sqrt(tf.reduce_mean(tf.square(
                tf.subtract(self.pred_inverse, self.y_inverse))))
        
        
        # choose for optimize. if gpu is available then choose gpu for optimize
        device = '/CPU:0'
        if tf.test.is_gpu_available():
            device = '/device:GPU:0'
            
        with tf.device(device):
            with tf.variable_scope('loss'):        
                self.loss = tf.reduce_mean(tf.square(tf.subtract(pred, self.y)))
                self.optimize = self.optimizer.minimize(self.loss)
        
    
    def step(self, sess, x, y, is_train=False):
        '''Feed input each step. Inputs is encoder_x, decoder_x, decoder_y.
        if is_train is set True then model is trained to optimize loss.
        Output is loss'''
        input_feed = {self.x: x,
                      self.y: y}
        
        
        output_feed = None
        
        if is_train:
            output_feed = [self.loss, self.optimize]
        else:
            output_feed = [self.loss]
        
        outputs = sess.run(output_feed, input_feed)
        
        return outputs[0]
    
    
    
    def multi_step(self, sess, X, Y, is_train=False):
        '''Feed through many batch size, each batch size corresponse step'''
        num_batches = 0
        total_loss = 0.0
        
        while True:
            x = X[num_batches * self.batch_size : 
                (num_batches+1) * self.batch_size]
            y = Y[num_batches * self.batch_size : 
                (num_batches+1) * self.batch_size]
                
            _loss = self.step(sess, x y, is_train)
            
            total_loss += _loss
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        return avg_loss
    
    
    def fit(self, train, val=None, test=None, folder_result=None, 
            config_name=None, verbose=1):
        history_file = None
        error_file = None
        predict_file = None
        model_file = None
        mae_rmse_file = None
        
        if folder_result and config_name:
            history_file = folder_result + config_name + '_history.png'
#            error_file = folder_result + config_name + '_error.csv'
#            predict_file = folder_result + config_name + '_predict.csv'
#            model_file = folder_result + config_name + '_model_encoder_decoder.ckpt'
            mae_rmse_file = folder_result + 'mae_rmse_log.csv'
            
        
        train_losses = []
        val_losses = []
        for epoch in range(self.num_epochs):
            loss_train = self.train(self.sess, train)
            train_losses.append(loss_train)
            
            if val:
                loss_val = self.validate(self.sess, val)
                val_losses.append(loss_val)
                
                if verbose == 1:
                    print('Epoch #%d loss train = %.7f  loss_val = %.7f' % (epoch,
                          loss_train, loss_val))
                    
                # apply early stop
                if early_stop(val_losses, self.patience):
                    print('finished training at epoch', epoch)
                    break
            else:
                if verbose == 1:
                    print('Epoch #%d loss train = %.7f' % (epoch,
                          loss_train))
        
        if val:
            if folder_result and config_name:
                log = {'train': train_losses, 'val': val_losses}
                df_log = pd.DataFrame(log)
                df_log.to_csv(error_file, index=None)
            
            plot_loss(train_losses, val_losses, history_file)
        else:
            plot_loss(train_losses)
            
        if test:
            self.test(sess, test, None, mae_rmse_file)
            
            
            
    def train(self, sess, data):
        X = data[0]
        Y = data[1]
        return self.multi_step(sess, X, Y, True)
    
    def validate(self, sess, data):
        X = data[0]
        Y = data[1]
        return self.multi_step(sess, X, Y, False)
    
    def test(self, sess, data, log_file=None, log_mae_rmse=None):
        X = data[0]
        Y = data[1]
        
        mae = []
        rmse = []
        num_batches = 0
        total_loss = 0.0
        
        predict = []
        actual = []
        
        while True:
            x = X[num_batches * self.batch_size : 
                (num_batches + 1) * self.batch_size]
                
            y = Y[num_batches * self.batch_size : 
                (num_batches + 1) * self.batch_size]
                
            input_feed = {self.x: x, 
                          self.y: y}
            
            output_feed = [self.pred_inverse, 
                            self.y_inverse, 
                            self.MAE, 
                            self.RMSE, 
                            self.loss]
                
            outputs = self.sess.run(output_feed, input_feed)
            
            mae.append(outputs[2])
            rmse.append(outputs[3])
            total_loss += outputs[4]
            num_batches += 1
            
            predict.extend(outputs[0][:, 0])
            actual.extend(outputs[1][:, 0])
        mae = np.mean(mae)
        rmse = np.mean(rmse)
        avg_loss = total_loss / num_batches
        print('loss: %.7f  mae: %.7f  rmse: %.7f' % (avg_loss, mae, rmse))
         
        with open(log_mae_rmse, 'a+') as f:
            f.write('%f, %f\n' % (mae, rmse))
            
        
#        log = {'predict': predict, 'actual': actual}
#        df_log = pd.DataFrame(log)
#        df_log.to_csv(log_file, index=None)