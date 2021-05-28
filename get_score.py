from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import os


def get_throughput_rmse():
    data_score = pd.DataFrame(columns=['transfer_id', 'rmse', 'mse', 'mae'])
    # data_score = pd.read_csv('result/throughput/mrp/mrp_score.csv')
    real_data = pd.read_csv('result/throughput/mrp/real/mrp/1144.csv')
    # real_data = pd.read_csv('data/real_time_data/mrp/collect/50/50_1112/70_480.csv')
    predict_data = pd.read_csv('result/throughput/mrp/predict/mrp/predict_1144.csv')
    metric = pd.read_csv('data/mrp_100g_data_normalize_metrics.csv')
    real_data['Goodput'] = real_data['Goodput'].apply(lambda x: x/metric['Goodput'].iloc[0])
    # real_data['Goodput'] = real_data['Goodput'].apply(lambda x: x / 8000000000)
    predict_data['predict_throughput'] = predict_data['predict_throughput'].apply(lambda x: x/metric['Goodput'].iloc[0])
    # predict_data['predict_throughput'] = predict_data['predict_throughput'].apply(lambda x: x / 8000000000)
    real_throughput = [real_data['Goodput'].iloc[8], real_data['Goodput'].iloc[16], real_data['Goodput'].iloc[24],
                       real_data['Goodput'].iloc[32]]
    predict_throughput = [predict_data['predict_throughput'].iloc[0], predict_data['predict_throughput'].iloc[1], predict_data['predict_throughput'].iloc[2],
                          predict_data['predict_throughput'].iloc[3]]

    throughput_rmse = mean_squared_error(real_throughput, predict_throughput,squared=False)
    throughpute_mse = mean_squared_error(real_throughput, predict_throughput)
    throughput_mae = mean_absolute_error(real_throughput, predict_throughput)
    data_score = data_score.append([{'transfer_id': 1144, 'rmse': throughput_rmse, 'mse': throughpute_mse, 'mae': throughput_mae}], ignore_index=True)
    print(throughput_rmse)
    print(throughpute_mse)
    print(throughput_mae)
    data_score.to_csv('result/throughput/mrp/mrp_score.csv',index=False)



if __name__ == '__main__':
    get_throughput_rmse()