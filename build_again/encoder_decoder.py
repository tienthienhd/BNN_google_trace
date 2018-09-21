#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 19:24:55 2018

@author: tienthien
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils


def rnn_cell(rnn_unit,
             layers_units, 
             activation='tanh', 
             input_keep_prob=1.0, 
             output_keep_prob=1.0, 
             state_keep_prob=1.0, 
             variational_recurrent=False):
    
    '''
    Args:

    input_keep_prob: unit Tensor or float between 0 and 1, input keep 
        probability; if it is constant and 1, no input dropout will be added.
    output_keep_prob: unit Tensor or float between 0 and 1, output keep 
        probability; if it is constant and 1, no output dropout will be added.
    state_keep_prob: unit Tensor or float between 0 and 1, output keep 
        probability; if it is constant and 1, no output dropout will be added. 
        State dropout is performed on the outgoing states of the cell. Note the 
        state components to which dropout is applied when state_keep_prob is 
        in (0, 1) are also determined by the argument dropout_state_filter_visitor 
        (e.g. by default dropout is never applied to the c component of 
        an LSTMStateTuple).
    variational_recurrent: Python bool. If True, then the same dropout pattern 
        is applied across all time steps per run call. If this parameter is set, 
        input_size must be provided.
    '''
    
    if rnn_unit == 'lstm':
        rnn_cell_type = tf.nn.rnn_cell.LSTMCell
    elif rnn_unit == 'gru':
        rnn_cell_type = tf.nn.rnn_cell.GRUCell
#    elif rnn_unit == 'rnn':
#        rnn_cell_type = tf.nn.rnn_cell.RNNCell
    else:
        raise Exception('Choose a valid RNN unit type.')
        
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
    
    
    
    cells = []
    for num_units in layers_units:
        cell = rnn_cell_type(num_units, activation=activation)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
                                 input_keep_prob=input_keep_prob, 
                                 output_keep_prob=output_keep_prob,
                                 state_keep_prob=state_keep_prob,
                                 variational_recurrent=variational_recurrent,
                                 dtype=tf.float32)
        cells.append(cell)
    return tf.nn.rnn_cell.MultiRNNCell(cells)
        

    

    

def get_state_variables(rnn_unit_type, batch_size, cell):
    # For each layer, get the initial state and make a variable out of it
    # to enable updating its value.
    state_variables = []
    if rnn_unit_type == 'lstm':
        for state_c, state_h in cell.zero_state(batch_size, tf.float32):
            state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                tf.Variable(state_c, trainable=False),
                tf.Variable(state_h, trainable=False)))
    else: 
        for state in cell.zero_state(batch_size, tf.float32):
            state_variables.append(tf.Variable(state, trainable=False))
    # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
    return tuple(state_variables)


def get_state_update_op(rnn_unit_type, state_variables, new_states):
    # Add an operation to update the train states with the last state tensors
    update_ops = []
    if rnn_unit_type == 'lstm':
        for state_variable, new_state in zip(state_variables, new_states):
            # Assign the new state to the state variables on this layer
            update_ops.extend([state_variable[0].assign(new_state[0]),
                               state_variable[1].assign(new_state[1])])
    else:
        for state_variable, new_state in zip(state_variables, new_states):
            # Assign the new state to the state variables on this layer
            update_ops.append(state_variable.assign(new_state))
    # Return a tuple in order to combine all update_ops into a single operation.
    # The tuple's actual value should not be used.
    return tf.tuple(update_ops)





    
    
class EncoderDecoder(object):
    def __init__(self, config=None, sess=None, max_min_data=None):
        tf.reset_default_graph()
        
        self.rnn_unit = config['rnn_unit_type']
        self.activation = config['activation']
        self.input_keep_prob = config['input_keep_prob']
        self.output_keep_prob = config['output_keep_prob']
        self.state_keep_prob = config['state_keep_prob']
        self.variational_recurrent = config['variational_recurrent']
        
        self.layers_units = config['layers_units']
        
        self.encoder_sliding = config['sliding_encoder']
        self.decoder_sliding = config['sliding_decoder']
        
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.num_features = len(config['features'][0])
        
        self.patience = config['patience']
        
        if max_min_data:
            self.max = max_min_data[0]
            self.min = max_min_data[1]
        
        self.build_model()
        
        if sess:
            self.sess = sess
        else:
            self.sess = tf.Session()
            
        self.sess.run(tf.global_variables_initializer())



    def build_model(self):
        '''Build model with hyperparameters got when create object model encoder
        decoder.
        Model include 2 part: encoder and decoder'''
        
        
        # placeholder for inputs
        self.encoder_x = tf.placeholder(dtype=tf.float32, 
                                        shape=[None, None, self.num_features], 
                                        name='encoder_x')
        
        self.decoder_x = tf.placeholder(dtype=tf.float32, 
                                        shape=[None, None, self.num_features], 
                                        name='decoder_x')
        
        self.decoder_y = tf.placeholder(dtype=tf.float32, 
                                        shape=[None, 1], 
                                        name='decoder_y')
    
        # encoder graph and function to update state of encoder
        with tf.variable_scope('encoder'):
            encoder_cell = rnn_cell(self.rnn_unit,
                                    self.layers_units,
                                    self.activation,
                                    self.input_keep_prob,
                                    self.output_keep_prob,
                                    self.state_keep_prob,
                                    self.variational_recurrent)
            
            # encoder state is variable but it is not trainable
            encoder_state = get_state_variables(self.rnn_unit, self.batch_size, 
                                           encoder_cell)
            
            # get output and state after feed inputs
            encoder_outputs, new_state = tf.nn.dynamic_rnn(cell=encoder_cell, 
                                                   inputs=self.encoder_x,
                                                   initial_state=encoder_state,
                                                   dtype=tf.float32)
            
            # update variables encoder state
            self.update_op = get_state_update_op(self.rnn_unit, encoder_state, 
                                                 new_state)
                        
            
            # identity for encoder outputs to load indivial encoder_outputs
            encoder_outputs = tf.identity(encoder_outputs, 
                                          name='encoder_outputs')
         
        # decoder graph and output dense layer
        with tf.variable_scope('decoder'):
            
            decoder_cell = rnn_cell(self.rnn_unit,
                                    self.layers_units,
                                    self.activation,
                                    self.input_keep_prob,
                                    self.output_keep_prob,
                                    self.state_keep_prob,
                                    self.variational_recurrent)
            
            # get output and state after feed inputs
            decoder_outputs, decoder_state = tf.nn.dynamic_rnn(cell=decoder_cell, 
                                                   inputs=self.decoder_x,
                                                   initial_state=encoder_state)
            
            # output dense layer
            pred_decoder = tf.layers.dense(inputs=decoder_outputs[:, -1, :], 
                                           units=1, 
                                           name='dense_output')      
            pred_decoder = tf.identity(pred_decoder, 'decoder_pred')
        
        # metrics to validate accuracy
        with tf.variable_scope('validate_predict'):
            self.pred_inverse = pred_decoder * (self.max + self.min) + self.min
            self.y_inverse = self.decoder_y * (self.max + self.min) + self.min
            
            self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.pred_inverse, 
                                                         self.y_inverse)))
            self.RMSE = tf.sqrt(tf.reduce_mean(tf.square(
                    tf.subtract(self.pred_inverse, self.y_inverse))))
        
        # choose for optimize. if gpu is available then choose gpu for optimize
        device = '/CPU:0'
        if tf.test.is_gpu_available():
            device = '/device:GPU:0'
        
        with tf.device(device):
            with tf.name_scope('loss_optimizer'):
                self.loss = tf.reduce_mean(tf.squared_difference(pred_decoder, 
                                                            self.decoder_y))
                self.optimizer  = tf.train.AdamOptimizer().minimize(self.loss)
        
        self.saver = tf.train.Saver()


    def step(self, encoder_x, decoder_x, decoder_y, is_train=False):
        '''Feed input each step. Inputs is encoder_x, decoder_x, decoder_y.
        if is_train is set True then model is trained to optimize loss.
        Output is loss'''
        input_feed = {self.encoder_x: encoder_x,
                      self.decoder_x: decoder_x,
                      self.decoder_y: decoder_y
                }
        
        output_feed = None
        if is_train:
            output_feed = [self.update_op, self.loss, self.optimizer]
        else:
            output_feed = [self.loss]
            
        outputs = self.sess.run(output_feed, input_feed)
        
        if is_train:
            return outputs[1]
        else:
            return outputs[0]
        
    def multi_step(self, encoder_x, decoder_x, decoder_y, is_train=False):
        '''Feed through many batch size, each batch size corresponse step'''
        encoder_x = utils.padding(encoder_x, self.batch_size)
        decoder_x = utils.padding(decoder_x, self.batch_size)
        decoder_y = utils.padding(decoder_y, self.batch_size)
        
        num_batches = int(len(encoder_x)/self.batch_size)
        if len(encoder_x) % self.batch_size != 0:
            num_batches += 1
        total_loss = 0.0
        
        for batch in range(num_batches):
            e_x = encoder_x[batch * self.batch_size : 
                (batch+1) * self.batch_size]
            d_x = decoder_x[batch * self.batch_size : 
                (batch+1) * self.batch_size]
            d_y = decoder_y[batch * self.batch_size : 
                (batch+1) * self.batch_size]
                
            _loss = self.step(e_x, d_x, d_y, is_train)
            total_loss += _loss
        avg_loss = total_loss / num_batches
        
        return avg_loss
        
                
                
#        num_batches = 0
#        total_loss = 0.0
#        
#        try:
#            while True:
#                e_x = encoder_x[num_batches * self.batch_size : 
#                    (num_batches+1) * self.batch_size]
#                d_x = decoder_x[num_batches * self.batch_size : 
#                    (num_batches+1) * self.batch_size]
#                d_y = decoder_y[num_batches * self.batch_size : 
#                    (num_batches+1) * self.batch_size]
#                    
#                _loss = self.step(e_x, d_x, d_y, is_train)
#                
#                total_loss += _loss
#                num_batches += 1
#        except tf.errors.InvalidArgumentError:
#            ''' if exception appear then this is last batch . The last batch not
#            enough examples to feed through graph because state of encoder is
#            fixed'''
#            pass
#        
#        avg_loss = total_loss / num_batches
#        
#        return avg_loss
    
    def train(self, train_set, val_set=None, show_step=10):
        train_encoder_x = train_set[0]
        train_decoder_x = train_set[1]
        train_decoder_y = train_set[2]
        
        val_encoder_x = None
        val_decoder_x = None
        val_decoder_y = None
        if val_set:
            val_encoder_x = train_set[0]
            val_decoder_x = train_set[1]
            val_decoder_y = train_set[2]
            
        train_losses = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            train_loss = self.multi_step(train_encoder_x, train_decoder_x, 
                                         train_decoder_y, True)
            train_losses.append(train_loss)
            
            if val_set:
                val_loss = self.multi_step(val_encoder_x, val_decoder_x, 
                                           val_decoder_y, False)
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
        encoder_x = test_set[0]
        decoder_x = test_set[1]
        decoder_y = test_set[2]
        
        mae = []
        rmse = []
        total_loss = 0.0
        num_batches = 0
        
        predict = []
        actual = []
        
        encoder_x = utils.padding(encoder_x, self.batch_size)
        decoder_x = utils.padding(decoder_x, self.batch_size)
        decoder_y = utils.padding(decoder_y, self.batch_size)
        
        num_batches = int(len(encoder_x)/self.batch_size)
        if len(encoder_x) % self.batch_size != 0:
            num_batches += 1
        total_loss = 0.0
        
        for batch in range(num_batches):
            e_x = encoder_x[batch * self.batch_size : 
                (batch+1) * self.batch_size]
            d_x = decoder_x[batch * self.batch_size : 
                (batch+1) * self.batch_size]
            d_y = decoder_y[batch * self.batch_size : 
                (batch+1) * self.batch_size]
                
            input_feed = {self.encoder_x: e_x,
                            self.decoder_x: d_x,
                            self.decoder_y: d_y}
            
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
        
        return predict, actual, mae, rmse
    
    def fit(self, dataset, log_name):
        history_img = log_name + '_history.png'
        predict_log = log_name + '_predict.csv'
        predict_log_img = log_name + '_predict.png'
        model_file = log_name + '_model_ed.ckpt'
        mae_rmse_log = log_name[0: log_name.rindex('/')+1] + 'mae_rmse_log.csv'
        
        dataset.prepare_data_inputs_encoder(self.encoder_sliding, self.decoder_sliding)
        train = dataset.get_data_encoder('train')
        val = dataset.get_data_encoder('val')
        test = dataset.get_data_encoder('test')
        
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
        
        
        