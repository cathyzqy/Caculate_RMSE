from datetime import datetime, timedelta
import requests
import time
import pandas as pd
import numpy
import extractor
import random
import os
import real_time_prediction
import N_prediction



def finish_transfer(transfer_id, orchestrator, sender, receiver):
    response = requests.post('{}/wait/{}'.format(orchestrator, transfer_id))
    result = response.json()
    # print(result)

    cleanup(sender, receiver)


def cleanup(sender, receiver, retry=5):
    for i in range(0, retry):
        response = requests.get('{}/cleanup/nuttcp'.format(sender))
        if response.status_code != 200: continue
        response = requests.get('{}/cleanup/nuttcp'.format(receiver))
        if response.status_code != 200: continue

        response = requests.get('{}/cleanup/stress'.format(sender))
        response = requests.get('{}/cleanup/stress'.format(receiver))
        response = requests.get('{}/cleanup/fio'.format(sender))
        response = requests.get('{}/cleanup/fio'.format(receiver))

        return
    raise Exception('Cannot cleanup after %s tries' % retry)


def get_transfer(transfer_id, orchestrator):
    response = requests.get('{}/transfer/{}'.format(orchestrator, transfer_id))
    result = response.json()
    print(result)


def parse_nvme_usage(filename):
    df = pd.read_csv(filename, parse_dates=[0])
    df['elapsed'] = (df['Time'] - df['Time'][0]) / numpy.timedelta64(1, 's')

    df = df[['elapsed', 'written_mean']].astype('int32').set_index('elapsed')
    df['written_mean'] = df['written_mean'].apply(str) + 'M'

    return df.to_dict()['written_mean']


def prepare_transfer(srcdir, sender, receiver):
    result = requests.get('{}/files/{}'.format(sender, srcdir))
    files = result.json()
    # files = files[0:5000]
    file_list = [srcdir + i['name'] for i in files if i['type'] == 'file']
    dirs = [srcdir + i['name'] for i in files if i['type'] == 'dir']

    response = requests.post('{}/create_dir/'.format(receiver), json=dirs)
    if response.status_code != 200:
        raise Exception('failed to create dirs')
    return file_list


def start_transfer_nvmeof(file_list, num_workers, orchestrator, sender_id, receiver_id, duration=5):
    data = {
        'srcfile': file_list,
        'dstfile': file_list,  # ['/dev/null'] * len(file_list),
        'num_workers': num_workers,
        'iomode': 'read',
        'blocksize': 1024
    }

    response = requests.post('{}/transfer/fio/{}/{}'.format(orchestrator, sender_id, receiver_id),
                             json=data)  # error out
    result = response.json()
    assert result['result'] == True
    transfer_id = result['transfer']
    return transfer_id

def start_transfer_mrp(file_list, num_workers, orchestrator, duration=5):
    data = {
        'srcfile': file_list,
        'dstfile': file_list,  # ['/dev/null'] * len(file_list),
        'num_workers': num_workers,
        'duration': duration  # ,
        # 'blocksize' : 8192
    }

    response = requests.post('{}/transfer/nuttcp/2/1'.format(orchestrator), json=data)  # error out
    result = response.json()
    assert result['result'] == True
    transfer_id = result['transfer']
    return transfer_id


def start_nvme_usage(nvme_usage, sender):
    data = {
        'sequence': nvme_usage,
        'file': 'disk0/fiotest',
        'size': '1G',
        'address': ''
    }
    response = requests.post('{}/receiver/stress'.format(sender), json=data)
    result = response.json()
    assert result.pop('result') == True


def wait_for_transfer(transfer_id, orchestrator, sender):
    while True:
        response = requests.get('{}/check/{}'.format(orchestrator, transfer_id))
        result = response.json()
        if result['Unfinished'] == 0:
            response = requests.get('{}/cleanup/stress'.format(sender))
            break
        time.sleep(30)


def get_realtime_data(sender, receiver, sender_instance, receiver_instance, start_time, sequence, monitor, cluster):
    df = extractor.export_data(sender, receiver, sender_instance, receiver_instance, start_time.timestamp(),
                               datetime.now().timestamp(), monitor, cluster)

    df = update_params(df, sequence)

    return df


def update_params(df, sequence):
    indices = sorted(sequence.keys())
    for i in range(0, len(indices)):
        next_t = indices[i + 1] if i < len(indices) - 1 else (df.index[-1] - df.index[0]).seconds
        for j in sequence[indices[i]].keys():
            if j not in df:
                df[j] = numpy.NaN
            df.loc[((df.index - df.index[0]).seconds >= indices[i]) & ((df.index - df.index[0]).seconds < next_t), j] = \
                sequence[indices[i]][j]
    return df


def set_limit():
    provisor = 'http://dtn-provisor.starlight.northwestern.edu'
    resp = requests.get('{}/connect'.format(provisor))
    data = {
        'port': 'et-0/0/31',
        'limit': 100,
        'vlan': 555
    }

    # resp = requests.delete('{}/limit'.format(provisor), json=data)
    resp = requests.post('{}/limit'.format(provisor), json=data)
    print(resp)


def dynamic_transfer(num_workers, sequence, orchestrator, sender, receiver, sender_instance,
                     receiver_instance, srcdir, monitor, cluster, sender_id, receiver_id):
    assert type(sequence) == dict

    nvme_usage = parse_nvme_usage('data/nvme_usage_daily.csv')
    sender_obj = requests.get('{}/DTN/{}'.format(orchestrator, 2)).json()
    receiver_obj = requests.get('{}/DTN/{}'.format(orchestrator, 1)).json()

    file_list = prepare_transfer(srcdir, sender, receiver)
    if cluster == 'mrp_nvmeof':
        transfer_id = start_transfer_nvmeof(file_list, num_workers, orchestrator, sender_id, receiver_id)
    else:
        transfer_id = start_transfer_mrp(file_list, num_workers, orchestrator)

    start_nvme_usage(nvme_usage, sender)

    start_time = datetime.now()
    print('transfer_id %s , start_time %s' % (transfer_id, start_time))

    collect_dir = 'data/real_time_data/'+str(cluster)+'/collect/' + str(sequence[0]['num_workers']) + '/'
    folder = os.path.exists(collect_dir)
    if not folder:
        os.makedirs(collect_dir)

    predict_dir = 'data/real_time_data/'+str(cluster)+'/predict/' + str(sequence[0]['num_workers']) + '/'
    folder_predict = os.path.exists(predict_dir)
    if not folder_predict:
        os.makedirs(predict_dir)

    folder_name = str(sequence[0]['num_workers']) + '_' + str(transfer_id)
    os.mkdir(os.path.join(collect_dir, folder_name))
    os.mkdir(os.path.join(predict_dir, folder_name))

    data_throughput = pd.DataFrame(columns=['time', 'predict_throughput'])
    if cluster == 'mrp_nvmeof':
        features = ['NVMe_total_util', 'CPU', 'Memory_used', 'NVMe_from_transfer', 'num_workers']
    else:
        features = ['NVMe_total_util', 'CPU', 'Memory_used', 'Goodput', 'num_workers']

    intervals = sorted(sequence.keys())
    for interval in intervals:
        if interval == 0:
            N = sequence[interval]['num_workers']
            continue
        # sleeping for the next change of N interval
        while interval > (datetime.now() - start_time).total_seconds():
            time.sleep(0.1)

        # getting real-time data from the start of the transfer to now.
        # You can update the sequence as you change dynamically after the transfer/id/scale api call
        print('elapsed_time %s, interval %s' % ((datetime.now() - start_time).total_seconds(), interval))
        predict_time = datetime.now()
        dataframe = get_realtime_data(sender_obj, receiver_obj, sender_instance, receiver_instance, start_time,
                                      sequence, monitor, cluster)
        dataframe.to_csv(collect_dir + folder_name + '/' + str(int(N)) + '_' + str(int(interval)-120) + '.csv', index=None)
        pred = pd.DataFrame(data=dataframe, columns=features)
        # print(pred)

        firststarttime = datetime.now()
        real_time_prediction.prediction(collect_dir + folder_name + '/' + str(int(N)) + '_' + str(int(interval)-120) + '.csv', cluster)
        firstendtime = datetime.now()
        firstuse = firstendtime - firststarttime
        print('first predition time:', firstuse.total_seconds(), 's')

        secondstarttime = datetime.now()
        # N, predict_throughput = G_prediction.get_N(predict_dir + folder_name + '/' + str(int(N)) + '_real.csv', cluster)
        N, predict_throughput = N_prediction.get_N(predict_dir + folder_name + '/' + str(int(N)) + '_' + str(int(interval)-120) + '_real.csv', cluster)
        secondendtime = datetime.now()
        seconduse = (secondendtime - secondstarttime).total_seconds()
        print('second predition time:', seconduse, 's')
        print('num_wokers:' + str(N))
        print('predict throughput: ', predict_throughput)
        data_throughput = data_throughput.append([{'time': predict_time, 'predict_throughput': predict_throughput}],
                                                 ignore_index=True)

        data = sequence[interval]
        sequence.pop(interval)
        sequence[interval] = {'num_workers': int(N)}
        data = sequence[interval]
        response = requests.post('{}/transfer/{}/scale'.format(orchestrator, transfer_id), json=data)
        # update sequence if it is not from the data.
        if response.status_code != 200:
            print('failed to change transfer parameter')
            break
        else:
            print('Changed the parameters to %s' % data)

    dataframe = get_realtime_data(sender_obj, receiver_obj, sender_instance, receiver_instance, start_time,
                                  sequence, monitor, cluster)
    dataframe.to_csv(collect_dir + folder_name + '/' + str(int(N)) + '_' + str(int(interval)) + '.csv', index=None)

    data_throughput.to_csv('result/throughput/' + str(cluster) + '/predict/' + str(cluster) + '/predict_' + str(transfer_id) + '.csv',
                           index=None)

    wait_for_transfer(transfer_id, orchestrator, sender)

    response = requests.get('{}/stress/poll'.format(sender), json={})

    return transfer_id


if __name__ == "__main__":
    cluster = 'mrp'

    if cluster == 'prp':
        orchestrator = 'https://dtn-orchestrator-2.nautilus.optiputer.net'
        sender = 'http://dtn-sender-2.nautilus.optiputer.net'
        receiver = 'http://dtn-receiver-2.nautilus.optiputer.net'
        srcdir = 'project/'
        sender_instance = 'siderea.ucsc.edu'
        receiver_instance = 'k8s-nvme-01.ultralight.org'
        monitor = 'https://thanos.nautilus.optiputer.net'
        sender_id = 2
        receiver_id = 1
    elif cluster == 'mrp':
        orchestrator = 'http://dtn-orchestrator.starlight.northwestern.edu'
        sender = 'http://dtn-sender.starlight.northwestern.edu'
        receiver = 'http://dtn-receiver.starlight.northwestern.edu'
        srcdir = 'project/'
        sender_instance = '165.124.33.175:9100'
        receiver_instance = '131.193.183.248:9100'
        monitor = 'http://165.124.33.158:9091/'
        sender_id = 2
        receiver_id = 1
    elif cluster == 'mrp_nvmeof':
        orchestrator = 'http://dtn-orchestrator.starlight.northwestern.edu'
        sender = 'http://dtn-sender.starlight.northwestern.edu'
        receiver = 'http://dtn-receiver-nvmeof.starlight.northwestern.edu'
        srcdir = 'project/'
        sender_instance = '165.124.33.175:9100'
        receiver_instance = '131.193.183.248:9100'
        monitor = 'http://165.124.33.158:9091/'
        sender_id = 2
        receiver_id = 4

    else:
        raise Exception('Only prp or mrp is supported')

    sequence = {  # time : parameters. Need to be updated dynamically for real-time data extraction
        0: {'num_workers': 50},
        120: {'num_workers': 50},
        240: {'num_workers': 50},
        360: {'num_workers': 50},
        480: {'num_workers': 50},
        # 600: {'num_workers': 50},
        # 720: {'num_workers': 50},

    }

    cleanup(sender, receiver)
    set_limit()
    transfer_id = dynamic_transfer(50, sequence, orchestrator, sender, receiver, sender_instance,
                                   receiver_instance, srcdir, monitor, cluster, sender_id, receiver_id)

    finish_transfer(transfer_id, orchestrator, sender, receiver)
    get_transfer(transfer_id, orchestrator)
    print(sequence)


