import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils

# import matplotlib
# matplotlib.use('Agg')


class MLP(object):
    def __init__(self, sess=None, config=None, max_min_data=None, model_encoder=None):
        tf.reset_default_graph()
        
        self.hidden_layers = config['hidden_layers']
        self.num_epochs = config['num_epochs']
        self.sliding_inference = int(config['sliding_inference'])

        self.activation = utils.activation(config['activation'])
        
        self.batch_size = int(config['batch_size'])
        self.optimizer = utils.optimizer(config['optimizer'], learning_rate=config['learning_rate'])
        self.dropout_rate = config['dropout_rate']
        self.patience = config['patience']
        
        if max_min_data:
            self.max = max_min_data[0]
            self.min = max_min_data[1]
        
        if sess:
            self.sess = sess
        else:
            self.sess = tf.Session()
        self.load_encoder_model(model_encoder)
        self.build_model()
        self.sess.run(tf.global_variables_initializer())

    def load_encoder_model(self, saved_file):
        print('Load encoder model:', saved_file)
        encoder_saver = tf.train.import_meta_graph(saved_file+'.meta')
        encoder_graph = tf.get_default_graph()
        self.encoder_x = encoder_graph.get_tensor_by_name('encoder_x:0')
        # print(self.encoder_x.shape)
        encoder_outputs = encoder_graph.get_tensor_by_name('encoder/encoder_outputs:0')
#        print(encoder_outputs.shape)
        output_encoder_sg = tf.stop_gradient(encoder_outputs)
#        print(output_encoder_sg.shape)
        self.encoder_last_outputs = output_encoder_sg[:, -1, :]
        self.encoder_last_outputs = tf.reshape(self.encoder_last_outputs,
                                               shape=(self.encoder_last_outputs.shape[0],
                                               self.encoder_last_outputs.shape[1],
                                               1))
        # print(self.encoder_last_outputs.shape)
        encoder_saver.restore(self.sess, saved_file)
        
    def build_model(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.sliding_inference, self.encoder_x.shape[2]], name='x')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')

        prev_layer = self.encoder_last_outputs[:self.batch_size]
        # print(prev_layer.shape)
        prev_layer = tf.concat(values=[prev_layer, self.x], axis=1)
        # print(prev_layer.shape)
        prev_layer = tf.reshape(prev_layer, (int(prev_layer.shape[0]), int(prev_layer.shape[1] * prev_layer.shape[2])))
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
        
    def step(self, x_encoder, x_mlp, y, is_train=False):
        '''Feed input each step. Inputs is encoder_x, decoder_x, decoder_y.
        if is_train is set True then model is trained to optimize loss.
        Output is loss'''
        x_encoder = utils.padding(x_encoder, batch_size=self.encoder_last_outputs.shape[0], pos=1)
        input_feed = {self.encoder_x: x_encoder,
                      self.x: x_mlp,
                      self.y: y,
                }
        
        output_feed = None
        if is_train:
            output_feed = [self.loss, self.optimize]
        else:
            output_feed = [self.loss]
        outputs = self.sess.run(output_feed, input_feed)
        return outputs[0]
    
    def multi_step(self, X_encoder, X_mlp, Y, is_train=False):
        '''Feed through many batch size, each batch size corresponse step'''
        num_batches = 0
        total_loss = 0.0
        X_encoder = utils.padding(X_encoder, self.batch_size)
        X_mlp = utils.padding(X_mlp, self.batch_size)
        Y = utils.padding(Y, self.batch_size)
        
        num_batches = int(len(X_encoder)/self.batch_size)
        if len(X_encoder) % self.batch_size != 0:
            num_batches += 1
        total_loss = 0.0
        
        for batch in range(num_batches):
            x1 = X_encoder[batch * self.batch_size :
                (batch+1) * self.batch_size]
            x2 = X_mlp[batch * self.batch_size :
                (batch+1) * self.batch_size]
            y = Y[batch * self.batch_size : 
                (batch+1) * self.batch_size]
                
            _loss = self.step(x1, x2, y, is_train)
            
            total_loss += _loss
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_set, val_set=None, show_step=10):
        train_x1 = train_set[0]
        train_x2 = train_set[1]
        train_y = train_set[2]
        
        val_x1 = None
        val_x2 = None
        val_x3 = None
        if val_set:
            val_x1 = val_set[0]
            val_x2 = val_set[1]
            val_y = val_set[2]
        
        train_losses = np.empty(self.num_epochs)
        val_losses = np.empty(self.num_epochs)
        
        for epoch in range(self.num_epochs):
            train_loss = self.multi_step(train_x1, train_x2, train_y, True)
            train_losses[epoch] = train_loss
            
            if val_set:
                val_loss = self.multi_step(val_x1, val_x2, val_y, False)
                val_losses[epoch] = val_loss
                
                if show_step is not 0 and epoch % show_step == 0:
                    print('Epoch #%d loss train = %.7f  loss_val = %.7f' % (epoch,
                              train_loss, val_loss))
                    
                # apply early stop
                if utils.early_stop(val_losses, patience=self.patience, idx=epoch):
                    print('Finished training config {} at epoch {}'.format('config_name', epoch))
                    train_losses = train_losses[:epoch]
                    val_losses = val_losses[:epoch]
                    break
            else:
                if show_step is not 0 and epoch % show_step == 0:
                    print('Epoch #%d loss train = %.7f' % (epoch,train_loss))
        return train_losses, val_losses
    
    def validate(self, test_set):
        X_encoder = test_set[0]
        X_mlp = test_set[1]
        Y = test_set[2]

        X_encoder = utils.padding(X_encoder, self.batch_size)
        X_mlp = utils.padding(X_mlp, self.batch_size)
        Y = utils.padding(Y, self.batch_size)
        
        num_batches = int(len(X_encoder)/self.batch_size)
        if len(X_encoder) % self.batch_size != 0:
            num_batches += 1
        total_loss = 0.0

        mae = np.empty(num_batches)
        rmse = np.empty(num_batches)

        predict = np.empty(len(Y))
        actual = np.empty(len(Y))
        
        for batch in range(num_batches):
            x_e = X_encoder[batch * self.batch_size:
                (batch+1) * self.batch_size]
            x_e = utils.padding(x_e, self.encoder_last_outputs.shape[0], pos=1)
            x_m = X_mlp[batch * self.batch_size:
                (batch+1) * self.batch_size]
            y = Y[batch * self.batch_size : 
                (batch+1) * self.batch_size]
                
            input_feed = {self.encoder_x: x_e,
                          self.x: x_m,
                          self.y: y}
            
            output_feed = [self.pred_inverse, 
                           self.y_inverse, 
                           self.MAE, 
                           self.RMSE, 
                           self.loss]
                
            outputs = self.sess.run(output_feed, input_feed)
            
            mae[batch] = outputs[2]
            rmse[batch] = outputs[3]
            total_loss += outputs[4]

            predict[batch*self.batch_size: (batch + 1)*self.batch_size] = outputs[0][:, 0]
            actual[batch*self.batch_size: (batch + 1)*self.batch_size] = outputs[1][:, 0]

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
        # print(mae_rmse_log)
        
        dataset.prepare_data_inputs_mlp(sliding_encoder=int(self.encoder_last_outputs.shape[1]),
                                        sliding_inference=self.sliding_inference)
        train = dataset.get_data_mlp('train')
        val = dataset.get_data_mlp('val')
        test = dataset.get_data_mlp('test')
        
        train_losses, val_losses = self.train(train, val, 50)
        losses_dict = {'train_loss': train_losses, 'val_loss': val_losses}
        utils.plot_log(losses_dict, ['epoch', 'loss'], history_img)
        
        predict, actual, mae, rmse = self.validate(test)
        # print(predict.shape, actual.shape)
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


utils.generate_config('test.json', 'log/test.csv', ['data', 'mlp'])
configs = utils.read_config('log/test.csv', chunksize=10)
config = None
for c in configs:
    for idx, co in c.iterrows():
        config = co

from data import Data
data = Data(config)
mlp = MLP(config=config, max_min_data=data.get_max_min(), model_encoder='./log/results_ed/0_model_ed.ckpt')
# data.prepare_data_inputs_mlp(sliding_inference=config['sliding_inference'], sliding_encoder=4)
 # a = data.get_data_mlp('train')
 # print(a[0].shape, a[1].shape, a[2].shape)
mlp.fit(data, 'log/')