import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from preprocessing_data import Data
import utils
import math


def load_model(sess, saved_model):
    print('Load model:', saved_model)
    saver = tf.train.import_meta_graph(saved_model + '.meta')
    graph = tf.get_default_graph()

    encoder_x = graph.get_tensor_by_name('encoder_x:0')
    mlp_x = graph.get_tensor_by_name('x:0')
    prediction = graph.get_tensor_by_name('prediction:0')

    saver.restore(sess, saved_model)
    #    writer = tf.summary.FileWriter('./log/', sess.graph)
    return encoder_x, mlp_x, prediction


def feed_forward(sess, model, inputs, dataset):
    x_encoder = model[0]
    x_mlp = model[1]
    prediction = model[2]

    x_e = inputs[0]
    x_m = inputs[1]

    pred = sess.run(prediction, feed_dict={x_encoder:x_e, x_mlp:x_m})
    pred_inv = dataset.denormalize(pred)
    # pred_inv = pred_inv[1:]
    return pred_inv

def inherent_noise(sess, model, val_set, dataset):
    pred = feed_forward(sess, model, val_set, dataset)
    actual_inv = dataset.denormalize(val_set[2])
    actual_inv = np.reshape(actual_inv, (-1, 1))
    # actual_inv = actual_inv[:-1]
    return np.mean(np.square(pred - actual_inv))

def mcdropout(sess, model, inputs, dataset, num_iterator=10):
    preds = np.empty(shape=(num_iterator, len(inputs[0])))
    for i in range(num_iterator):
        pred = feed_forward(sess, model, inputs, dataset)
        preds[i] = np.reshape(pred, len(pred))
    y_mc = np.mean(preds, axis=0)
    theta = [[np.square(y - p) for p in preds[:, 0]] for y in y_mc]
    theta = np.array(theta)
    theta = np.mean(theta, axis=1)
    return y_mc, theta

def inference(sess, model, inputs, dataset, theta2_sq):
    y_mc, theta1_sq = mcdropout(sess, model, inputs, dataset)

    theta = np.sqrt(theta1_sq + theta2_sq)
    return y_mc, theta

def plot_inference(actual, predict, theta):
    plt.fill_between(range(len(actual)), predict + theta, predict - theta, color='c')
    plt.plot(predict, 'b--', label='prediction')
    plt.plot(actual, 'r-', label='actual')
    plt.legend()
    plt.show()



tf.reset_default_graph()
sess = tf.Session()
x_encoder, x_mlp, prediction = load_model(sess, 'results/mlp/1_model_mlp.ckpt')
print(x_encoder.shape, x_mlp.shape, prediction.shape)

configs_read = utils.read_config('log/results_mlp/configs_mlp.csv', 1000, 0)
configs = [a for a in configs_read]
configs = configs[0]
config = configs.iloc[9]
# print(config)

dataset = Data('data/data_resource_usage_5Minutes_6176858948.csv')
train, val, test = dataset.get_data(30, 4)

# predict = feed_forward(sess, (x_encoder, x_mlp, prediction), test, dataset)

n22 = inherent_noise(sess, (x_encoder, x_mlp, prediction), (val[0][:10], val[1][:10], val[2][:10]), dataset)
ymc, n12 = mcdropout(sess, (x_encoder, x_mlp, prediction), (test[0][:10], test[1][:10], test[2][:10]), dataset)
print('theta 2 =', n22)
print('theta 1 =', n12)


ymc, theta = inference(sess, (x_encoder, x_mlp, prediction), (test[0][:1000], test[1][:1000], test[2][:1000]), dataset, n22)

# ymc = ymc[1:]
# theta = theta[1:]
# print('y =', ymc)
# print('theta =', theta)
# print(test[2][:10])
a = dataset.denormalize(test[2][:1000])
# a = a[:-1]

plot_inference(a,ymc, theta)