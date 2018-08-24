#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:15:24 2018

@author: tienthien
"""

import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt



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


class EncoderDecoder(object):
    
    def __init__(self, config=None, max_min_data=None):
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
        self.num_features = config['num_features']
        
        self.patience = config['patience']
        
        if max_min_data:
            self.max = max_min_data[0]
            self.min = max_min_data[1]
        
        self.build_model()
    
        
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
                
    
    def step(self, sess, encoder_x, decoder_x, decoder_y, is_train=False):
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
            
        outputs = sess.run(output_feed, input_feed)
        
        if is_train:
            return outputs[1]
        else:
            return outputs[0]
    
    
    def multi_step(self, sess, encoder_x, decoder_x, decoder_y, is_train=False):
        '''Feed through many batch size, each batch size corresponse step'''
        num_batches = 0
        total_loss = 0.0
        try:
            while True:
                e_x = encoder_x[num_batches * self.batch_size : 
                    (num_batches+1) * self.batch_size]
                d_x = decoder_x[num_batches * self.batch_size : 
                    (num_batches+1) * self.batch_size]
                d_y = decoder_y[num_batches * self.batch_size : 
                    (num_batches+1) * self.batch_size]
                    
                _loss = self.step(sess, e_x, d_x, d_y, is_train)
                
                total_loss += _loss
                num_batches += 1
        except tf.errors.InvalidArgumentError:
            ''' if exception appear then this is last batch . The last batch not
            enough examples to feed through graph because state of encoder is
            fixed'''
            pass
        
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
        
#        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            train_losses = []
            val_losses = []
            
            for epoch in range(self.num_epochs):
                loss_train = self.train(sess, train)
                train_losses.append(loss_train)
                
                if val:
                    loss_val = self.validate(sess, val)
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
#                if folder_result and config_name:
#                    log = {'train': train_losses, 'val': val_losses}
#                    df_log = pd.DataFrame(log)
#                    df_log.to_csv(error_file, index=None)
                
                plot_loss(train_losses, val_losses, history_file)
            else:
                plot_loss(train_losses)
                
            if test:
                self.test(sess, test, None, mae_rmse_file)
            
#            saver.save(sess, model_file)
    
    
    def train(self, sess, data):
        encoder_x = data[0]
        decoder_x = data[1]
        decoder_y = data[2]

        return self.multi_step(sess, encoder_x, decoder_x, decoder_y, True)
    
    def validate(self, sess, data):
        encoder_x = data[0]
        decoder_x = data[1]
        decoder_y = data[2]
        
        return self.multi_step(sess, encoder_x, decoder_x, decoder_y, False)
    
    def test(self, sess, data, log_file=None, log_mae_rmse=None):
        encoder_x = data[0]
        decoder_x = data[1]
        decoder_y = data[2]
        
        mae = []
        rmse = []
        total_loss = 0.0
        num_batches = 0
        
        predict = []
        actual = []
        try:
            while True:
                e_x = encoder_x[num_batches * self.batch_size : 
                    (num_batches+1) * self.batch_size]
                d_x = decoder_x[num_batches * self.batch_size : 
                    (num_batches+1) * self.batch_size]
                d_y = decoder_y[num_batches * self.batch_size : 
                    (num_batches+1) * self.batch_size]
                    
                    
                feed_input = {self.encoder_x: e_x,
                                self.decoder_x: d_x,
                                self.decoder_y: d_y}
                
                feed_output = [self.pred_inverse, 
                               self.y_inverse, 
                               self.MAE, 
                               self.RMSE, 
                               self.loss]
                    
                outputs = sess.run(feed_output, feed_input)
                
                mae.append(outputs[2])
                rmse.append(outputs[3])
                total_loss += outputs[4]
                num_batches += 1
                
                predict.extend(outputs[0][:, 0])
                actual.extend(outputs[1][:, 0])
        except tf.errors.InvalidArgumentError:
            pass
        mae = np.mean(mae)
        rmse = np.mean(rmse)
        avg_loss = total_loss / num_batches
        print('loss: %.7f  mae: %.7f  rmse: %.7f' % (avg_loss, mae, rmse))
    
        if log_mae_rmse:
            with open(log_mae_rmse, 'a+') as f:
                f.write('%f, %f\n' % (mae, rmse))
            
#        if log_file:
#            log = {'predict': predict, 'actual': actual}
#            df_log = pd.DataFrame(log)
#            df_log.to_csv(log_file, index=None)



############################# Test ######################################

def test():
    from config import Config
    from data import Data
    config = Config('config.json')
    data = Data(config.data)
    
    # prepare data
    data.prepare_data_inputs_encoder(config.encoder_decoder['sliding_encoder'],
                                config.encoder_decoder['sliding_decoder'])
    train = data.get_data_encoder('train')
    val = data.get_data_encoder('val')
    test = data.get_data_encoder('test')
    
    
    #e_x = np.random.randn(100, 4, 1)
    #d_x = np.random.randn(100, 4, 1)
    #d_y = np.random.randn(100, 1)
    #train = val = test = (e_x, d_x, d_y)
    
#    print(data.get_max_min())
    a = EncoderDecoder(config.encoder_decoder, data.get_max_min())
    a.fit(train, val, test, './log/', 'config')
    #a = rnn_cell('gru', [2, 4])
    #b = a.zero_state(8, tf.float32)
    #
    #
    #sess = tf.Session()
    #
    #c = sess.run(b)
    #for t in c:
    #    print(t)
    #    print('=============')
    #    


#test()














  
        
        
        
        
        
        
        
        
        