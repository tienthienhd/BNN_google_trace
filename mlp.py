import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils
import math


import matplotlib
matplotlib.use('Agg')


class MLP(object):
    def __init__(self, encoder_model, hidden_layers, activation, optimizer, dropout_rate, batch_size, learning_rate, sliding):
        tf.reset_default_graph()
        self.batch_size = batch_size
        self.sess = tf.Session()
        self.load_encoder_model(encoder_model)
        self.build_model(hidden_layers=hidden_layers,
                         activation=activation,
                         dropout_rate=dropout_rate,
                         optimizer=optimizer,
                         learning_rate=learning_rate,
                         sliding=sliding)
        self.sess.run(tf.global_variables_initializer())

    def load_encoder_model(self, saved_file):
        print('Load encoder model:', saved_file)
        encoder_saver = tf.train.import_meta_graph(saved_file + '.meta')
        encoder_graph = tf.get_default_graph()
        self.encoder_x = encoder_graph.get_tensor_by_name('encoder_x:0')
        encoder_outputs = encoder_graph.get_tensor_by_name('encoder/encoder_outputs:0')
        output_encoder_sg = tf.stop_gradient(encoder_outputs)
        self.encoder_last_outputs = output_encoder_sg[:, -1, :]
        # print(self.encoder_last_outputs.shape)
        self.encoder_last_outputs = tf.reshape(self.encoder_last_outputs,
                                               shape=(-1,
                                                      self.encoder_last_outputs.shape[1],
                                                      1))
        # self.batch_size = int(self.encoder_last_outputs.shape[0])
        self.sliding_encoder = int(self.encoder_last_outputs.shape[1])
        encoder_saver.restore(self.sess, saved_file)

    def build_model(self, hidden_layers, activation, dropout_rate, optimizer, learning_rate, sliding):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, sliding, self.encoder_x.shape[2]],
                                name='x')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')

        # prev_layer = self.encoder_last_outputs
        # print(prev_layer.shape)
        prev_layer = tf.concat(values=[self.encoder_last_outputs, self.x], axis=1)
        # print(prev_layer.shape)
        prev_layer = tf.reshape(prev_layer, (-1, int(prev_layer.shape[1] * prev_layer.shape[2])))
        for i, num_units in enumerate(hidden_layers):
            prev_layer = tf.layers.dense(inputs=prev_layer,
                                         activation=activation,
                                         units=num_units,
                                         name='layer' + str(i))
            prev_layer = tf.layers.dropout(inputs=prev_layer,
                                           rate=dropout_rate)

        self.pred = tf.layers.dense(inputs=prev_layer,
                               units=1,
                               name='output_layer')
        self.pred = tf.identity(self.pred, 'prediction')

        # choose for optimize. if gpu is available then choose gpu for optimize
        device = '/CPU:0'
        if tf.test.is_gpu_available():
            device = '/device:GPU:0'

        with tf.device(device):
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.square(tf.subtract(self.pred, self.y)))
                self.optimize = utils.optimizer(optimizer, learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    def step(self, x_encoder, x_mlp, y, mode=2):
        '''

        :param x_encoder:
        :param x_mlp:
        :param y:
        :param mode: 0: training; 1: validate; 2:predict
        :return:
        '''
        if mode == 0:
            input_feed = {self.encoder_x: x_encoder,
                          self.x: x_mlp,
                          self.y: y,
                          }
            output_feed = [self.loss, self.optimize]
        elif mode == 1:
            input_feed = {self.encoder_x: x_encoder,
                          self.x: x_mlp,
                          self.y: y,
                          }
            output_feed = [self.loss]
        elif mode == 2:
            input_feed = {self.encoder_x: x_encoder,
                          self.x: x_mlp
                          }
            output_feed = [self.pred]

        outputs = self.sess.run(output_feed, input_feed)
        return outputs[0]

    def multi_step(self, x_encoder, x_mlp, y, mode=2):
        if mode == 0:
            num_batches = int(len(x_encoder) / int(self.batch_size))
            if len(x_encoder) % self.batch_size != 0:
                num_batches += 1
            total_loss = 0.0
            for batch in range(num_batches):
                x1 = x_encoder[batch * self.batch_size:
                               (batch + 1) * self.batch_size]
                x2 = x_mlp[batch * self.batch_size:
                           (batch + 1) * self.batch_size]
                y_ = y[batch * self.batch_size:
                       (batch + 1) * self.batch_size]
                total_loss += self.step(x1, x2, y_, mode=0) * len(x1)
            return total_loss / len(x_encoder)
        elif mode == 1:
            return self.step(x_encoder, x_mlp, y, mode=1)
        elif mode == 2:
            return self.step(x_encoder, x_mlp, None, mode=2)

    def train(self, train_set, val_set=None, num_epochs=100, patience=20, show_step=10):
        train_x1 = train_set[0]
        train_x2 = train_set[1]
        train_y = train_set[2]

        if val_set:
            val_x1 = val_set[0]
            val_x2 = val_set[1]
            val_y = val_set[2]

        train_losses = np.empty(num_epochs)
        val_losses = np.empty(num_epochs)

        for epoch in range(num_epochs):
            train_loss = self.multi_step(train_x1, train_x2, train_y, mode=0)
            train_losses[epoch] = train_loss

            if val_set:
                val_loss = self.multi_step(val_x1, val_x2, val_y, mode=1)
                val_losses[epoch] = val_loss

                if show_step is not 0 and epoch % show_step == 0:
                    print('Epoch #%d loss train = %.7f  loss_val = %.7f' % (epoch,
                                                                            train_loss, val_loss))

                # apply early stop
                if utils.early_stop(val_losses, patience=patience, idx=epoch):
                    print('Finished training config {} at epoch {}'.format('config_name', epoch))
                    train_losses = train_losses[:epoch]
                    val_losses = val_losses[:epoch]
                    break
            else:
                if show_step is not 0 and epoch % show_step == 0:
                    print('Epoch #%d loss train = %.7f' % (epoch, train_loss))
            if math.isnan(train_loss):
                print('Early stop because loss is nan')
                break
        return train_losses, val_losses

    def validate(self, test_set, dataset):
        x_encoder = test_set[0]
        x_mlp = test_set[1]
        y = test_set[2]

        prediction = self.multi_step(x_encoder, x_mlp, y, mode=2)

        prediction = dataset.denormalize(prediction)
        y_actual = dataset.denormalize(y)

        # prediction = prediction[1:]
        # y_actual = y_actual[:-1]

        mae = np.mean(np.abs(prediction-y_actual))
        rmse = np.sqrt(np.mean(np.square(prediction-y_actual)))
        print('mae: %.7f  rmse: %.7f' % (mae, rmse))
        return prediction, y_actual, mae, rmse

    def fit(self, dataset, num_epochs, patience, sliding, log_name):
        history_img = log_name + '_history.png'
        predict_log = log_name + '_predict.csv'
        predict_log_img = log_name + '_predict.png'
        model_file = log_name + '_model_mlp.ckpt'
        mae_rmse_log = log_name[0: log_name.rindex('/') + 1] + 'mae_rmse_log.csv'

        train, val, test = dataset.get_data(sliding_encoder=self.sliding_encoder, sliding_2=sliding)

        train_losses, val_losses = self.train(train, val,
                                              num_epochs=num_epochs, patience=patience, show_step=50)
        losses_dict = {'train_loss': train_losses, 'val_loss': val_losses}
        utils.plot_log(losses_dict, ['epoch', 'loss'], history_img)

        predict, actual, mae, rmse = self.validate(test, dataset)
        predict = predict[:, 0]
        actual = actual[:, 0]
        test_dict = {'predict': predict, 'actual': actual}
        utils.plot_log(test_dict, file_save=predict_log_img)

        df_test = pd.DataFrame(test_dict)
        df_test.to_csv(predict_log, index=False)

        self.saver.save(self.sess, model_file)

        # with open(mae_rmse_log, 'a+') as f:
        #     f.write('%f, %f\n' % (mae, rmse))
        with open(mae_rmse_log, 'a+') as f:
            f.write('%s,%f,%f\n' % (log_name[log_name.rindex('/')+1:],mae, rmse))

    def predict(self, inputs):
        pass

    def close_sess(self):
        self.sess.close()


# import preprocessing_data
# dataset = preprocessing_data.Data('data/data_resource_usage_5Minutes_6176858948.csv')
# mlp = MLP('results/ed/1_model_ed.ckpt', [32, 16, 4], 'tanh', 'rmsprop', 0.1, 16, 0.001, 4)
# mlp.fit(dataset, num_epochs=1000, patience=5, sliding=4, log_name='results/mlp/1')

# train, val, test = dataset.get_data(30, 4)
# mlp.validate(test, dataset)