import csv
import pandas as pd
from pandas import read_csv
import math
import numpy as np
import pickle
from datetime import datetime
import tracemalloc
from time import process_time


def load_data(file_path):
    dataset = read_csv(file_path)
    dataset.dropna(axis=0, how='any', inplace=True)

    return dataset


def add_num(datapath, datasave):
    dataadd = pd.read_csv(datapath)

    datanew = pd.DataFrame(columns=['NVMe_total_util', 'CPU', 'Memory_used','num_workers'])

    i = 4
    num = 1
    while (i < 404):
        for j in range(0, 4):
            datanew = datanew.append([{'NVMe_total_util': dataadd['NVMe_total_util'].loc[j], 'CPU': dataadd['CPU'].loc[j],
                             'Memory_used': dataadd['Memory_used'].loc[j], 'num_workers': num/100}], ignore_index=True)
            i = i + 1

        num = num + 1

    datanew.to_csv(datasave, index=None)


def get_N(real_data, cluster):
    path = real_data.split('.')
    path_adddata = path[0] + 'a.csv'
    add_num(real_data, path_adddata)
    xpred = load_data(path_adddata)

    # model = pickle.load(open('model/XGBoost/'+str(cluster)+'/xg_predict_n.dat', "rb"))
    model = pickle.load(open('model/XGBoost/mrp/xgnt_mrp_100_new_metric.dat', "rb"))
    metric = pd.read_csv('data/'+str(cluster)+'_100g_data_normalize_metrics.csv')

    y_pred = model.predict(xpred)

    index = np.array([i for i in range(0, 400)])
    newfind = np.c_[y_pred, index]

    findmax = newfind[np.argsort(-newfind[:, 0])]

    max_index = np.where(findmax[:, 0] == np.amax(findmax[:, 0]))[0][0]
    max_num = findmax[max_index, 1]
    xgb_max_predmin = xpred.loc[max_num]['num_workers']

    num_workers = int(xgb_max_predmin * 100)
    throughput = findmax[0, 0] * metric.iloc[0][3]

    return num_workers, throughput


if __name__ == '__main__':
    cluster = 'mrp'
    get_N('data/real_time_data/'+str(cluster)+'/predict/30/30_1045/74_real.csv',cluster)
