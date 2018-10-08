#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:20:50 2018

@author: tienthien
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils

#import matplotlib
#matplotlib.use('Agg')
    
class MLP(object):
    def __init__(self, sess=None, config=None, max_min_data=None, type_model=None):
        tf.reset_default_graph()
        
        self.hidden_layers = config['hidden_layers']
        activation = config['activation']
        self.num_epochs = config['num_epochs']
        self.input_dim = int(config['input_dim'])
        if activation == 'tanh':
            activation = tf.nn.tanh
        elif activation == 'sigmoid':
            activation = tf.nn.sigmoid
        elif activation == 'relu':
            activation = tf.nn.relu
        elif activation == 'elu':
            activation = tf.nn.elu
        else:
            raise Exception('Choose a valid activation')
        self.activation = activation
        
        self.batch_size = int(config['batch_size'])
        self.optimizer = tf.train.AdamOptimizer()
        self.dropout_rate = config['dropout_rate']
        self.patience = config['patience']
        
#        type_model = 'mem_multivariate_ed'
#        print(type_model)
#        self.num_features = 
        
        if max_min_data:
            self.max = max_min_data[0]
            self.min = max_min_data[1]
        
        encoder_saved_file = './log/models/' + type_model + '.ckpt'
        
        if sess:
            self.sess = sess
        else:
            self.sess = tf.Session()
        self.load_encoder_model(encoder_saved_file)
        self.build_model()
        self.sess.run(tf.global_variables_initializer())
        
        
        
    def load_encoder_model(self, saved_file):
        print('Load encoder model:', saved_file)
        encoder_saver = tf.train.import_meta_graph(saved_file+'.meta')
        encoder_graph = tf.get_default_graph()
        self.encoder_x = encoder_graph.get_tensor_by_name('encoder_x:0')
#        print(self.encoder_x.shape)
        encoder_outputs = encoder_graph.get_tensor_by_name('encoder/encoder_outputs:0')
#        print(encoder_outputs.shape)
        output_encoder_sg = tf.stop_gradient(encoder_outputs)
#        print(output_encoder_sg.shape)
        self.encoder_last_outputs = output_encoder_sg[:, -1, :]
        encoder_saver.restore(self.sess, saved_file)
        
        
    def build_model(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='x')
        self.y= tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')
        
        prev_layer = self.encoder_last_outputs # can concatinate with self.x
#        print(prev_layer.shape)
        for i, num_units in enumerate(self.hidden_layers):
            prev_layer = tf.layers.dense(inputs=prev_layer, 
                                         activation=self.activation,
                                         units=num_units,
                                         name='layer'+str(i))
            prev_layer = tf.layers.dropout(inputs=prev_layer,
                                           rate=self.dropout_rate)
            
        pred = tf.layers.dense(inputs=prev_layer,
                               units=1,
                               name='output_layer')
        
        self.pred_inverse = pred * (self.max + self.min) + self.min
        self.pred_inverse = tf.identity(self.pred_inverse, name='prediction')
        self.y_inverse = self.y * (self.max + self.min) + self.min
        
        self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.pred_inverse,
                                                     self.y_inverse)))
        self.RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(
                self.pred_inverse, self.y_inverse))))
        
        # choose for optimize. if gpu is available then choose gpu for optimize
        device = '/CPU:0'
        if tf.test.is_gpu_available():
            device = '/device:GPU:0'
            
        with tf.device(device):
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.square(tf.subtract(pred, self.y)))
                self.optimize = self.optimizer.minimize(self.loss)
                
        self.saver = tf.train.Saver()
        
    def step(self, x, y, is_train=False):
        '''Feed input each step. Inputs is encoder_x, decoder_x, decoder_y.
        if is_train is set True then model is trained to optimize loss.
        Output is loss'''
        input_feed = {self.encoder_x: x,
                      self.y: y,
                }
        
        output_feed = None
        if is_train:
            output_feed = [self.loss, self.optimize]
        else:
            output_feed = [self.loss]
        outputs = self.sess.run(output_feed, input_feed)
        return outputs[0]
    
    def multi_step(self, X, Y, is_train=False):
        '''Feed through many batch size, each batch size corresponse step'''
        num_batches = 0
        total_loss = 0.0
        X = utils.padding(X, self.batch_size)
        Y = utils.padding(Y, self.batch_size)
        
        num_batches = int(len(X)/self.batch_size)
        if len(X) % self.batch_size != 0:
            num_batches += 1
        total_loss = 0.0
        
        for batch in range(num_batches):
            x = X[batch * self.batch_size : 
                (batch+1) * self.batch_size]
            y = Y[batch * self.batch_size : 
                (batch+1) * self.batch_size]
                
            _loss = self.step(x, y, is_train)
            
            total_loss += _loss
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_set, val_set=None, show_step=10):
        train_x = train_set[0]
        train_y = train_set[1]
        
        val_x = None
        val_y = None
        if val_set:
            val_x = val_set[0]
            val_y = val_set[1]
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            train_loss = self.multi_step(train_x, train_y, True)
            train_losses.append(train_loss)
            
            if val_set:
                val_loss = self.multi_step(val_x, val_y, False)
                val_losses.append(val_loss)
                
                if show_step is not 0 and epoch % show_step == 0:
                    print('Epoch #%d loss train = %.7f  loss_val = %.7f' % (epoch,
                              train_loss, val_loss))
                    
                # apply early stop
                if utils.early_stop(val_losses, self.patience):
                    print('Finished training config {} at epoch {}'.format('config_name', epoch))
                    break
            else:
                if show_step is not 0 and epoch % show_step == 0:
                    print('Epoch #%d loss train = %.7f' % (epoch,train_loss))
        return train_losses, val_losses
    
    def validate(self, test_set):
        X = test_set[0]
        Y = test_set[1]
        
        mae = []
        rmse = []
        total_loss = 0.0
        num_batches = 0
        
        predict = []
        actual = []
        X = utils.padding(X, self.batch_size)
        Y = utils.padding(Y, self.batch_size)
        
        num_batches = int(len(X)/self.batch_size)
        if len(X) % self.batch_size != 0:
            num_batches += 1
        total_loss = 0.0
        
        for batch in range(num_batches):
            x = X[batch * self.batch_size : 
                (batch+1) * self.batch_size]
            y = Y[batch * self.batch_size : 
                (batch+1) * self.batch_size]
                
            input_feed = {self.encoder_x: x,
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
            
            predict.extend(outputs[0][:, 0])
            actual.extend(outputs[1][:, 0])
        mae = np.mean(mae)
        rmse = np.mean(rmse)
        avg_loss = total_loss / num_batches
        print('loss: %.7f  mae: %.7f  rmse: %.7f' % (avg_loss, mae, rmse))
        
        return predict, actual, mae, rmse
        
    def fit(self, dataset, log_name):
        history_img = log_name + '_history.png'
        predict_log = log_name + '_predict.csv'
        predict_log_img = log_name + '_predict.png'
        model_file = log_name + '_model_mlp.ckpt'
        mae_rmse_log = log_name[0: log_name.rindex('/')+1] + 'mae_rmse_log.csv'
        
        dataset.prepare_data_inputs_mlp(input_dim=self.input_dim)
        train = dataset.get_data_mlp('train')
        val = dataset.get_data_mlp('val')
        test = dataset.get_data_mlp('test')
        
        train_losses, val_losses = self.train(train, val, 50)
        losses_dict = {'train_loss': train_losses, 'val_loss': val_losses}
        utils.plot_log(losses_dict, ['epoch', 'loss'], history_img)
        
        predict, actual, mae, rmse = self.validate(test)
        test_dict = {'predict': predict, 'actual': actual}
        utils.plot_log(test_dict, file_save=predict_log_img)
        
        df_test = pd.DataFrame(test_dict)
        df_test.to_csv(predict_log, index=False)
        
        self.saver.save(self.sess, model_file)
        
        with open(mae_rmse_log, 'a+') as f:
            f.write('%f, %f\n' % (mae, rmse))
    def predict(self, inputs):
        pass
    
    def close_sess(self):
        self.sess.close()
        
