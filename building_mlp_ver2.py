# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 00:55:56 2018

@author: tienthien
"""

import tensorflow as tf
import numpy as np


class MLP(object):
    
    def __init__(self, config=None, sess=None):
#        self.hidden_layers = config['hidden_layers']
#        self.activation = tf.nn.tanh
#        self.num_epochs = config['num_epochs']
#        
#        self.batch_size = config['batch_size']
#        self.optimizer = tf.train.AdamOptimizer()
#        self.dropout_rate = 0.9

        self.hidden_layers = [2]
        self.activation = tf.nn.tanh
        self.num_epochs = 2
        
        self.batch_size = 32
        self.optimizer = tf.train.AdamOptimizer()
        self.dropout_rate = 0.9
        
        if not sess:
            self.sess = tf.Session()
        else:
            self.sess = sess
    
    def load_encoder(self):
        encoder_saver = tf.train.import_meta_graph('./log/model/result_config_0_model_encoder_decoder.ckpt.meta')
        encoder_graph = tf.get_default_graph()
        self.encoder_x = encoder_graph.get_tensor_by_name('encoder_x:0')
        encoder_outputs = encoder_graph.get_tensor_by_name('encoder/encoder_outputs:0')
        output_encoder_sg = tf.stop_gradient(encoder_outputs)
        self.encoder_last_outputs = output_encoder_sg[:, :, -1]
        encoder_saver.restore(self.sess, './log/model/result_config_0_model_encoder_decoder.ckpt')
        
    
    def build_model(self):
        self.load_encoder();
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
    
    def step(self, x, y, is_train=False):
        input_feed = {self.x: x,
                      self.y: y}
        output_feed = None
        if is_train:
            output_feed = [self.loss, self.optimize]
        else:
            output_feed = [self.loss]
        outputs = self.sess.run(output_feed, input_feed)
        return outputs[0]
    
    def save(self, file_name='mlp.ckpt'):
        saver = tf.train.Saver()
        saver.save(self.sess)
        pass
    
    def restore(self):
        pass
    

mlp = MLP()
mlp.build_model()