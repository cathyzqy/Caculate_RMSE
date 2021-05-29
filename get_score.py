from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import xgboost as xgb


def get_real_throughput_rmse_mrp():
    data_score = pd.DataFrame(columns=['transfer_id', 'rmse', 'mse', 'mae'])
    real_file_path = 'result/throughput/mrp/real/mrp_add/'
    predict_file_path = 'result/throughput/mrp/predict/mrp_add/'
    files = os.listdir(real_file_path)
    for file in files:
        transfer_id = file.split('.')[0]
        real_path = real_file_path + file
        real_data = pd.read_csv(real_path)
        predict_data = pd.read_csv(predict_file_path + 'predict_' + file)
        metric = pd.read_csv('data/mrp_100g_data_normalize_metrics.csv')
        real_data['Goodput'] = real_data['Goodput'].apply(lambda x: x / metric['Goodput'].iloc[0])
        predict_data['predict_throughput'] = predict_data['predict_throughput'].apply(
            lambda x: x / metric['Goodput'].iloc[0])
        real_throughput = [real_data['Goodput'].iloc[8], real_data['Goodput'].iloc[16], real_data['Goodput'].iloc[24],
                           real_data['Goodput'].iloc[32]]
        predict_throughput = [predict_data['predict_throughput'].iloc[0], predict_data['predict_throughput'].iloc[1],
                              predict_data['predict_throughput'].iloc[2],
                              predict_data['predict_throughput'].iloc[3]]

        throughput_rmse = mean_squared_error(real_throughput, predict_throughput, squared=False)
        throughpute_mse = mean_squared_error(real_throughput, predict_throughput)
        throughput_mae = mean_absolute_error(real_throughput, predict_throughput)
        data_score = data_score.append(
            [{'transfer_id': transfer_id, 'rmse': throughput_rmse, 'mse': throughpute_mse, 'mae': throughput_mae}],
            ignore_index=True)
        print(throughput_rmse)
        print(throughpute_mse)
        print(throughput_mae)

    data_score.to_csv('result/throughput/mrp/mrp_score_add.csv', index=False)


def get_real_throughput_rmse_prp():
    data_score = pd.DataFrame(columns=['transfer_id', 'rmse', 'mse', 'mae'])
    real_file_path = 'result/throughput/prp/real/prp_add/'
    predict_file_path = 'result/throughput/prp/predict/prp_add/'
    files = os.listdir(real_file_path)
    for file in files:
        transfer_id = file.split('.')[0]
        real_path = real_file_path + file
        real_data = pd.read_csv(real_path)
        predict_data = pd.read_csv(predict_file_path + 'predict_' + file)
        metric = pd.read_csv('data/mrp_100g_data_normalize_metrics.csv')
        real_data['Goodput'] = real_data['Goodput'].apply(lambda x: x / metric['Goodput'].iloc[0])
        predict_data['predict_throughput'] = predict_data['predict_throughput'].apply(
            lambda x: x / metric['Goodput'].iloc[0])
        real_throughput = [real_data['Goodput'].iloc[8], real_data['Goodput'].iloc[16], real_data['Goodput'].iloc[24],
                           real_data['Goodput'].iloc[32]]
        predict_throughput = [predict_data['predict_throughput'].iloc[0], predict_data['predict_throughput'].iloc[1],
                              predict_data['predict_throughput'].iloc[2],
                              predict_data['predict_throughput'].iloc[3]]

        throughput_rmse = mean_squared_error(real_throughput, predict_throughput, squared=False)
        throughpute_mse = mean_squared_error(real_throughput, predict_throughput)
        throughput_mae = mean_absolute_error(real_throughput, predict_throughput)
        data_score = data_score.append(
            [{'transfer_id': transfer_id, 'rmse': throughput_rmse, 'mse': throughpute_mse, 'mae': throughput_mae}],
            ignore_index=True)
        print(throughput_rmse)
        print(throughpute_mse)
        print(throughput_mae)

    data_score.to_csv('result/throughput/prp/prp_score_add.csv', index=False)


def get_train_throughput_rmse_mrp():
    train_data = pd.read_csv('data/train_data/mrp/mrp_xgboost.csv')
    features = ['NVMe_total_util', 'CPU', 'Memory_used', 'num_workers']
    output = ['Goodput']
    x_pred = train_data[features]
    y = train_data[output]
    booster1 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                                colsample_bytree=1, max_depth=7)

    train_X, test_X, train_y, test_y = train_test_split(x_pred, y, test_size=0.2, random_state=0)
    booster1.fit(train_X, train_y)
    prediction = booster1.predict(test_X)

    rmse = mean_squared_error(test_y, prediction, squared=False)
    mae = mean_absolute_error(prediction, test_y)
    mse = mean_squared_error(prediction, test_y)
    print('MRP training RMSE:', rmse)
    print('MRP training MSE:', mse)
    print('MRP training MAE:', mae)


def get_train_throughput_rmse_prp():
    train_data = pd.read_csv('data/train_data/prp/prp_xgboost.csv')
    features = ['NVMe_total_util', 'CPU', 'Memory_used', 'num_workers']
    output = ['Goodput']
    x_pred = train_data[features]
    y = train_data[output]
    booster1 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                                colsample_bytree=1, max_depth=7)

    train_X, test_X, train_y, test_y = train_test_split(x_pred, y, test_size=0.2, random_state=0)
    booster1.fit(train_X, train_y)
    prediction = booster1.predict(test_X)

    rmse = mean_squared_error(test_y, prediction, squared=False)
    mae = mean_absolute_error(prediction, test_y)
    mse = mean_squared_error(prediction, test_y)
    print('MRP training RMSE:', rmse)
    print('MRP training MSE:', mse)
    print('MRP training MAE:', mae)




if __name__ == '__main__':
    get_real_throughput_rmse_mrp()
    get_real_throughput_rmse_prp()
    get_train_throughput_rmse_mrp()
    get_train_throughput_rmse_prp()
