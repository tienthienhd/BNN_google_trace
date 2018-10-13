import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils

import matplotlib

matplotlib.use('Agg')


def rnn_cell(rnn_unit,
             layers_units,
             activation='tanh',
             input_size=1,
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

    activation = utils.activation(activation)

    cells = []
    for num_units in layers_units:
        cell = rnn_cell_type(num_units, activation=activation)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                             input_keep_prob=input_keep_prob,
                                             output_keep_prob=output_keep_prob,
                                             state_keep_prob=state_keep_prob,
                                             variational_recurrent=variational_recurrent,
                                             input_size=input_size,
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
    def __init__(self, unit_type, activation, layers_units, input_keep_prob,
                 output_keep_prob, state_keep_prob, variational_recurrent,
                 optimizer, batch_size,
                 num_features, learning_rate):
        tf.reset_default_graph()
        self.batch_size = batch_size
        self.build_model(rnn_unit_type=unit_type,
                         activation=activation,
                         layers_units=layers_units,
                         input_keep_prob=input_keep_prob,
                         output_keep_prob=output_keep_prob,
                         state_keep_prob=state_keep_prob,
                         variational_recurrent=variational_recurrent,
                         optimizer=optimizer,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         num_features=num_features)

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

    def build_model(self, rnn_unit_type, activation, layers_units, input_keep_prob,
                 output_keep_prob, state_keep_prob, variational_recurrent,
                 optimizer, batch_size,
                 num_features, learning_rate):
        '''Build model with hyperparameters got when create object model encoder
        decoder.
        Model include 2 part: encoder and decoder'''

        # placeholder for inputs
        self.encoder_x = tf.placeholder(dtype=tf.float32,
                                        shape=[None, None, num_features],
                                        name='encoder_x')

        self.decoder_x = tf.placeholder(dtype=tf.float32,
                                        shape=[None, None, num_features],
                                        name='decoder_x')

        self.decoder_y = tf.placeholder(dtype=tf.float32,
                                        shape=[None, 1],
                                        name='decoder_y')

        # encoder graph and function to update state of encoder
        with tf.variable_scope('encoder'):
            encoder_cell = rnn_cell(rnn_unit_type,
                                    layers_units,
                                    activation,
                                    num_features,
                                    input_keep_prob,
                                    output_keep_prob,
                                    state_keep_prob,
                                    variational_recurrent)

            # encoder state is variable but it is not trainable
            encoder_state = get_state_variables(rnn_unit_type, batch_size,
                                                encoder_cell)

            # get output and state after feed inputs
            encoder_outputs, new_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                           inputs=self.encoder_x,
                                                           initial_state=encoder_state,
                                                           dtype=tf.float32)

            # update variables encoder state
            self.update_op = get_state_update_op(rnn_unit_type, encoder_state,
                                                 new_state)

            # identity for encoder outputs to load indivial encoder_outputs
            encoder_outputs = tf.identity(encoder_outputs,
                                          name='encoder_outputs')

        # decoder graph and output dense layer
        with tf.variable_scope('decoder'):
            decoder_cell = rnn_cell(rnn_unit_type,
                                    layers_units,
                                    activation,
                                    input_keep_prob,
                                    output_keep_prob,
                                    state_keep_prob,
                                    variational_recurrent)

            # get output and state after feed inputs
            decoder_outputs, decoder_state = tf.nn.dynamic_rnn(cell=decoder_cell,
                                                               inputs=self.decoder_x,
                                                               initial_state=encoder_state)

            # output dense layer
            pred_decoder = tf.layers.dense(inputs=decoder_outputs[:, -1, :],
                                           units=1,
                                           activation=activation,
                                           name='dense_output')
            self.pred_decoder = tf.identity(pred_decoder, 'decoder_pred')

        # choose for optimize. if gpu is available then choose gpu for optimize
        device = '/CPU:0'
        if tf.test.is_gpu_available():
            device = '/device:GPU:0'

        with tf.device(device):
            with tf.name_scope('loss_optimizer'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.pred_decoder,
                                                                 self.decoder_y))
                self.optimizer = utils.optimizer(optimizer, learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    def step(self, encoder_x, decoder_x, decoder_y=None, mode=2):
        '''

        :param encoder_x:
        :param decoder_x:
        :param decoder_y:
        :param mode: 0: training; 1: validate; 2:predict
        :return:
        '''
        if mode == 0:
            input_feed = {self.encoder_x: encoder_x,
                          self.decoder_x: decoder_x,
                          self.decoder_y: decoder_y
                          }
            output_feed = [self.update_op, self.loss, self.optimizer]
        elif mode == 1:
            input_feed = {
                self.encoder_x: encoder_x,
                self.decoder_x: decoder_x,
                self.decoder_y: decoder_y
            }
            output_feed = [self.loss]
        elif mode == 2:
            input_feed = {self.encoder_x: encoder_x,
                          self.decoder_x: decoder_x
                          }
            output_feed = [self.pred_decoder]

        outputs = self.sess.run(output_feed, input_feed)

        if mode == 0:
            return outputs[1]
        elif mode == 1:
            return outputs[0]
        elif mode == 2:
            return outputs[0]

    def multi_step(self, encoder_x, decoder_x, decoder_y=None, mode=2):
        '''

        :param encoder_x:
        :param decoder_x:
        :param decoder_y:
        :param mode: 0: training; 1: validate; 2:predict
        :return:
        '''
        len_x = len(encoder_x)
        encoder_x = utils.padding(encoder_x, self.batch_size)
        decoder_x = utils.padding(decoder_x, self.batch_size)
        if mode != 2:
            decoder_y = utils.padding(decoder_y, self.batch_size)

        num_batches = int(len(encoder_x) / self.batch_size)
        if len(encoder_x) % self.batch_size != 0:
            num_batches += 1
        if mode == 0 or mode == 1:
            total_loss = 0.0
        elif mode == 2:
            prediction = np.empty((len(encoder_x), 1))

        for batch in range(num_batches):
            e_x = encoder_x[batch * self.batch_size:
                            (batch + 1) * self.batch_size]
            d_x = decoder_x[batch * self.batch_size:
                            (batch + 1) * self.batch_size]
            if mode == 0 or mode == 1:
                d_y = decoder_y[batch * self.batch_size:
                                (batch + 1) * self.batch_size]
            elif mode == 2:
                d_y = None
            output = self.step(e_x, d_x, d_y, mode)
            if mode == 0 or mode == 1:
                total_loss += output
            elif mode == 2:
                prediction[batch * self.batch_size:
                           (batch + 1) * self.batch_size] = output
        if mode == 0 or mode == 1:
            return total_loss / num_batches
        elif mode == 2:
            return prediction[len(encoder_x) - len_x:]

    def train(self, train_set, val_set=None, num_epochs=100, patience=20, show_step=10):
        train_encoder_x = train_set[0]
        train_decoder_x = train_set[1]
        train_decoder_y = train_set[2]

        if val_set:
            val_encoder_x = train_set[0]
            val_decoder_x = train_set[1]
            val_decoder_y = train_set[2]

        train_losses = np.zeros(num_epochs)
        val_losses = np.zeros(num_epochs)

        for epoch in range(num_epochs):
            train_loss = self.multi_step(train_encoder_x, train_decoder_x,
                                         train_decoder_y, mode=0)
            train_losses[epoch] = train_loss

            if val_set:
                val_loss = self.multi_step(val_encoder_x, val_decoder_x,
                                           val_decoder_y, mode=1)
                val_losses[epoch] = val_loss

                if show_step is not 0 and epoch % show_step == 0:
                    print('Epoch #%d loss train = %.7f  loss_val = %.7f' % (epoch,
                                                                            train_loss, val_loss))

                # apply early stop
                # print(utils.early_stop(val_losses, self.patience))
                if utils.early_stop(val_losses, idx=epoch, patience=patience):
                    print('Finished training config {} at epoch {}'.format('config_name', epoch))
                    train_losses = train_losses[:epoch]
                    val_losses = val_losses[:epoch]
                    break
            else:
                if show_step is not 0 and epoch % show_step == 0:
                    print('Epoch #%d loss train = %.7f' % (epoch, train_loss))
        return train_losses, val_losses

    def validate(self, test_set, dataset):
        encoder_x = test_set[0]
        decoder_x = test_set[1]
        decoder_y = test_set[2]

        prediction = self.multi_step(encoder_x, decoder_x, mode=2)

        prediction = dataset.denormalize(prediction)
        y_actual = dataset.denormalize(decoder_y)
        mae = np.mean(np.abs(prediction - y_actual))
        rmse = np.sqrt(np.mean(np.square(prediction-y_actual)))
        print('mae: %.7f  rmse: %.7f' % (mae, rmse))
        return prediction, y_actual, mae, rmse

    def fit(self, dataset, num_epochs, patience, encoder_sliding, decoder_sliding, log_name):
        history_img = log_name + '_history.png'
        predict_log = log_name + '_predict.csv'
        predict_log_img = log_name + '_predict.png'
        model_file = log_name + '_model_ed.ckpt'
        mae_rmse_log = log_name[0: log_name.rindex('/') + 1] + 'mae_rmse_log.csv'

        train, val, test = dataset.get_data(sliding_encoder=encoder_sliding, sliding_2=decoder_sliding)

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

        with open(mae_rmse_log, 'a+') as f:
            f.write('%f, %f\n' % (mae, rmse))

    def predict(self, inputs):
        pass

    def close_sess(self):
        self.sess.close()

#
# import preprocessing_data
# dataset = preprocessing_data.Data('data/data_resource_usage_5Minutes_6176858948.csv')
# train, val, test = dataset.get_data(4, 2)
# ed = EncoderDecoder('lstm', 'tanh', [4], 0.95, 0.95, 0.95, True, 'adam', 32, 1, 0.001)
# train_losses, val_losses = ed.train(train, val, 1, 10, 10)
#
# import matplotlib.pyplot as plt
# # plt.plot(train_losses, label='train_loss')
# # plt.plot(val_losses, label='val_loss')
# # plt.legend()
# # plt.show()
#
# prediction, y_actual, mae, rmse = ed.validate(test, dataset)
# test_dict = {'predict': prediction, 'actual': y_actual}
# utils.plot_log(test_dict)
# # print(pred)
