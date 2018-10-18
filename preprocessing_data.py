import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n...t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1,...t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with Nan values
    if(dropnan):
        agg.dropna(inplace=True)
    return agg


def split_data(data, val_size=0.2, test_size=0.2):
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data[:ntest]) * (1 - val_size)))

    train = data[:nval]
    val = data[nval:ntest]
    test = data[ntest:]
    return train, val, test


class Data(object):
    def __init__(self, data_path, full_features=None, use_features=None):
        '''

        :param data_path:
        :param full_features:
        :param use_features: Features use in model. The first feature is output feature
        '''

        if full_features is None:
            full_features = ["time_stamp", "numberOfTaskIndex", "numberOfMachineId",
                     "meanCPUUsage", "canonical_memory_usage", "AssignMem",
                     "unmapped_cache_usage", "page_cache_usage", "max_mem_usage",
                     "mean_diskIO_time", "mean_local_disk_space", "max_cpu_usage",
                     "max_disk_io_time", "cpi", "mai", "sampling_portion",
                     "agg_type", "sampled_cpu_usage"]
        if use_features is None:
            use_features = ["meanCPUUsage"]

        df = pd.read_csv(data_path, header=None, names=full_features, usecols=use_features)
        # df.plot()
        # plt.show()

        min_features = np.amin(df)
        max_features = np.amax(df)

        df_normalized = (df - min_features) / (max_features - min_features)
        # print(type(df_normalized))

        # min_features = dict()
        # max_features = dict()
        # dict_normalized = dict()
        # for feature in use_features:
        #     data = df.loc[:, feature]
        #     min_features[feature] = np.amin(data)
        #     max_features[feature] = np.amax(data)
        #     dict_normalized[feature] = (data - min_features[feature]) / (max_features[feature] - min_features[feature])
        #
        # df_normalized = pd.DataFrame(dict_normalized)

        self.min_features = min_features
        self.max_features = max_features
        self.df_normalized = df_normalized
        self.use_features = use_features
        self.num_features = len(use_features)

    def denormalize(self, data, feature=None):
        if feature is None:
            feature = self.use_features[0]
        return data * (self.max_features[feature] - self.min_features[feature]) + self.min_features[feature]

    def get_data(self, sliding_encoder, sliding_2):
        sliding = max(sliding_encoder, sliding_2)
        data = series_to_supervised(self.df_normalized, sliding, 1)
        # print(data.head())

        input_encoder = data.iloc[:, -(sliding_encoder+1)*self.num_features:-self.num_features].values
        input_decoder = data.iloc[:, -(sliding_2+1)*self.num_features:-self.num_features].values
        output = data.iloc[:, -self.num_features].values

        input_encoder = np.reshape(input_encoder, [input_encoder.shape[0], sliding_encoder, self.num_features])
        input_decoder = np.reshape(input_decoder, [input_decoder.shape[0], sliding_2, self.num_features])
        output = np.reshape(output, [output.shape[0], 1])
        # print(input_encoder.shape, input_decoder.shape, output.shape)

        train_e, val_e, test_e = split_data(input_encoder)
        train_d, val_d, test_d = split_data(input_decoder)
        train_o, val_o, test_o = split_data(output)
        # print(train_e[0])
        # print('=============')
        # print(train_d[0])
        # print('==============')
        # print(train_o[0])

        return (train_e, train_d, train_o), (val_e, val_d, val_o), (test_e, test_d, test_o)



data = Data('data/data_resource_usage_5Minutes_6176858948.csv', use_features=['meanCPUUsage', 'canonical_memory_usage'])
# train, val, test = data.get_data(4, 3)
# print(train[0][0], train[1][0], train[2][0])
# print("==========================")
# print(train[0][1], train[1][1], train[2][1])
