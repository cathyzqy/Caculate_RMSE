# -*- coding: utf-8 -*-
from math import sqrt
from numpy import concatenate
import numpy as np
# from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers
import tensorflow.keras
import keras
import pickle
from datetime import datetime
import math
import os
import time
from tensorflow.keras.models import model_from_json
import tracemalloc
from time import process_time


# NUM_EXPAND = 1024 * 1024
# pynvml.nvmlInit()
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def series_to_supervised(data, columns, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    dataline = df.shape[0]
    # Choose the most recent period of time
    databegin = dataline - n_in - n_out + 1
    cols, names = list(), list()
    line = list()
    agg = list()
    result = pd.DataFrame()
    # input sequence (t-n, ... t-1), shift means data moves i rows
    for i in range(n_in, 0, -1):
        names += [('%s(t-%d)' % (columns[j], i)) for j in range(n_vars)]
    for i in range(0, n_out):
        for j in range(0, n_in):
            line = line + df.iloc[j + i + databegin, :].tolist()
        agg.append(line)
        line = []
    result = pd.DataFrame(agg)

    result.columns = names

    return result


def load_data(file_path):
    dataset = read_csv(file_path)
    dataset.dropna(axis=0, how='any', inplace=True)

    return dataset


def normalize_and_make_series(dataset, look_back):
    # values = dataset.values
    # values = values.astype('float64')
    # # normalize features
    # features_predict = ['NVMe_total_util', 'CPU', 'Memory_used']
    # y_values = dataset[features_predict].values
    # # frame as supervised learning
    # column_num = dataset.columns.size
    # column_names = dataset.columns.tolist()
    # reframed = series_to_supervised(values, column_names, look_back, 4)
    # return reframed
    values = dataset.values
    values = values.astype('float64')
    # normalize features
    features_predict = ['NVMe_total_util','CPU', 'Memory_used']
    y_values = dataset[features_predict].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    scaled_y = scaler.fit_transform(y_values)
    # frame as supervised learning
    column_num = dataset.columns.size
    column_names = dataset.columns.tolist()
    reframed = series_to_supervised(scaled, column_names, look_back, 4)
    return reframed, scaler


def split_data(dataset, reframed, look_back):
    column_num = dataset.columns.size
    values = reframed.values
    test_X = values[:, :]

    test_X = test_X.reshape(test_X.shape[0], look_back, column_num)

    return test_X


def prediction(file_path, cluster):
    name = file_path.split('/')
    num = name[-1].split('.')
    testdata_path = 'data/real_time_data/'+str(cluster)+'/predict/' + name[-3] + '/' + name[-2] + '/' + str(num[0]) + '_real'
    # modelread_path = 'model/BI-LSTM/'+str(cluster)+'/increase_model/increase_train_model/50model.h5'
    modelread_path = 'model/BI-LSTM/mrp/model.h5'
    normalize_metric = pd.read_csv('data/'+str(cluster)+'_100g_data_normalize_metrics.csv')
    look_back = 5

    dataset = load_data(file_path)
    data = pd.DataFrame(dataset)
    if cluster == 'mrp_nvmeof':
        features = ['NVMe_total_util', 'CPU', 'Memory_used', 'NVMe_from_transfer', 'num_workers']
    else:
        features = ['NVMe_total_util', 'CPU', 'Memory_used', 'Goodput', 'num_workers']

    data = data[features]
    data[features[0]] = data[features[0]].apply(lambda x: x / normalize_metric[features[0]].iloc[0])
    data[features[1]] = data[features[1]].apply(lambda x: x / normalize_metric[features[1]].iloc[0])
    data[features[2]] = data[features[2]].apply(lambda x: x / normalize_metric[features[2]].iloc[0])
    data[features[3]] = data[features[3]].apply(lambda x: x / normalize_metric[features[3]].iloc[0])
    data[features[4]] = data[features[4]].apply(lambda x: x / normalize_metric[features[4]].iloc[0])

    data = data.iloc[-8:, :]
    print(data.head())

    reframed,scaler = normalize_and_make_series(data, look_back)
    test_X = split_data(data, reframed, look_back)

    model = load_model(modelread_path)

    predict = model.predict(test_X, 8)
    inv_yhat = np.c_[predict]
    inv_yhat = scaler.inverse_transform(inv_yhat)

    col = ['NVMe_total_util', 'CPU', 'Memory_used']
    pred = pd.DataFrame(data=inv_yhat, columns=col)
    print(pred)
    pred.to_csv(testdata_path + '.csv', index=None)

    pred['NVMe_total_util'] = pred['NVMe_total_util'].apply(lambda x: x * normalize_metric['NVMe_total_util'].iloc[0])
    pred['CPU'] = pred['CPU'].apply(lambda x: x * normalize_metric['CPU'].iloc[0])
    pred['Memory_used'] = pred['Memory_used'].apply(lambda x: x * normalize_metric['Memory_used'].iloc[0])
    print(pred)


if __name__ == '__main__':
    prediction('data/real_time_data/mrp/collect/30/30_1045/74.csv','mrp')
