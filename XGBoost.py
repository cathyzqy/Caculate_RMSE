import shap
import math
import re
import os
import csv
from pandas import read_csv
from prometheus_http_client import Prometheus
import pandas as pd
import numpy as np
import xgboost as xgb
from http.client import HTTPException
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import pickle
import time


def xgb_train(x, y):
    booster1 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                                colsample_bytree=1, max_depth=7)

    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)

    booster1.fit(train_X, train_y)

    model_socre = booster1.score(test_X, test_y)
    pickle.dump(booster1, open('model/XGBoost/'+str(cluster)+'/xgnt_mrp_100_new_metric.dat', "wb"))

    explainer = shap.TreeExplainer(booster1)
    shap_values = explainer.shap_values(test_X)
    shap.summary_plot(shap_values, test_X, plot_type="bar")

    prediction = booster1.predict(test_X)

    score_mae = mean_absolute_error(prediction, test_y)
    score_mse = mean_squared_error(prediction, test_y)
    rmse_score = mean_squared_error(test_y, prediction, squared=False)

    return rmse_score


if __name__ == '__main__':
    cluster = 'mrp'
    start_time = time.time()
    path_modeldata = 'data/historical_data/'+str(cluster)+'/100g_xgboost_data/xgboost.csv'
    data = pd.read_csv(path_modeldata)
    if cluster == 'mrp_nvmeof':
        features = ['NVMe_total_util', 'CPU', 'Memory_used', 'num_workers']
        output = ['NVMe_from_transfer']
    else:
        features = ['NVMe_total_util', 'CPU', 'Memory_used', 'num_workers']
        output = ['Goodput']

    x_s1 = data[features]
    y1 = data[output]
    xgb_score = xgb_train(x_s1, y1)
    end_time = time.time()
    print(xgb_score)
    print('whole time:' + str(end_time - start_time))
