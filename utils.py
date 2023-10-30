# -*- coding:utf-8 -*-
import copy
import csv
import json
import os
from collections import defaultdict
import numpy as np
import h5py as h5
import pandas
import torch
from torch.utils.data import DataLoader

from data_container import riverFlowDataset
from models.sts import STSGCN
from sklearn.metrics import mean_absolute_error, mean_squared_error
from numpy import random

import datetime
import networkx as nx


def formatTimeKey(dt):
    return dt.strftime("%Y-%-m-%-d %-H:%M")


def formatTime(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M")


def construct_model(config):
    if config["name"] == "sts":
        module_type = config['module_type']  # 模型类型
        act_type = config['act_type']  # GLU与Relu
        temporal_emb = config['temporal_emb']
        spatial_emb = config['spatial_emb']
        use_mask = config['use_mask']
        num_of_vertices = config['num_of_vertices']
        num_of_features = config['num_of_features']
        seq_input = config['seq_input']
        num_for_predict = config['num_for_predict']
        adj_filename = config['adj_filename']
        id_filename = config['id_filename']
        day = config["day"]
        ab = None
        if config.get("ab") != None:
            ab = config.get("ab")
        if id_filename is not None:
            if not os.path.exists(id_filename):
                id_filename = None

        adj_mx = construct_adj_matrix(id_filename, adj_filename, num_of_vertices, day, ab = ab)
        adj_mx = torch.from_numpy(adj_mx)
        print("The shape of localized adjacency matrix: {}".format(
            adj_mx.shape), flush=True)

        filters = config['filters']
        first_layer_embedding_size = config['embedding_size']
        net = STSGCN(adj_mx, seq_input, num_of_vertices, num_of_features, num_for_predict, filters
                     , module_type, act_type, use_mask, temporal_emb, spatial_emb,window_size=day,
                     embedding_size=first_layer_embedding_size)
    elif config["name"] == "fclstm":
        input_size = config["input_size"]
        hidden_size = config["hidden_size"]
        output_dim = config["output_dim"]
        num_layers = config["num_layers"]

        net = FCLSTM(input_size, hidden_size, output_dim, num_layers)
    elif config["name"] == "IndRNN":
        input_size = config["input_size"]
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        act = config["activate"]
        net = IndRNNv2(input_size, hidden_size, num_layers, batch_first=True, nonlinearity=act)
    elif config["name"] == "stgcn":
        feat_size = config["input_size"]
        hidden_size = config["hidden_size"]
        seq = config["seq_input"]
        num_of_vertices = config['num_of_vertices']
        adj_filename = config['adj_filename']
        id_filename = config['id_filename']
        adj_mx = construct_adj_matrix(id_filename, adj_filename, num_of_vertices,
                                      model="stgcn", cor=config["cor"])
        net = STGCN(feat_size, 5, hidden_size, adj_mx)
    elif config['name'] == "gcn":
        feat_size = config["input_size"]
        adj_filename = config['adj_filename']
        seq = config["seq_input"]
        id_filename = config['id_filename']
        num_of_vertices = config['num_of_vertices']
        adj_mx = construct_adj_matrix(id_filename, adj_filename, num_of_vertices,
                                      model="gcn")
        pred = config["num_for_predict"]
        net = baseLine.GCN(seq, pred, feat_size, adj_mx)
    else:
        raise Exception("没有实现该模型")
    return net


def construct_adj_matrix(id_filename, distance_df_filename, num_of_vertices, day=None,
                         type_='connectivity', model='sts', cor=None, ab=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix 当id_filename为None时返回0元素矩阵

    '''
    with open(id_filename, 'r') as f:
        id_dict = {i: idx
                   for idx, i in enumerate(f.read().strip().split('\n'))}
    import csv
    if model == "sts":
        num_of_vertices = int(num_of_vertices)
        A = np.zeros([num_of_vertices * day] * 2)
        Isdelay = True
        Isoneway =True
        if ab !=None:
            if ab.get("delay") != None:
                Isdelay = False
            if ab.get("oneway") != None:
                Isoneway = False
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 4:
                    continue
                i, j, delay, distance = row[0], row[1], row[2], float(row[3])
                if len(delay.split("-")) == 2:
                    upper_bound, low_bound = delay.split("-")  # 滞后天数的上下界
                    upper_bound = min(int(upper_bound), day - 1)
                    if type_ == 'connectivity':
                        for m in range(day):
                            try:
                                if Isdelay == True:
                                    A[id_dict[j] + (m + upper_bound) * num_of_vertices, id_dict[
                                        i] + m * num_of_vertices] = 1  # 选择T时刻(位于矩阵的左上角)作为起点 往右是T+1 往下是T+1
                                    if Isoneway == False:
                                        A[id_dict[i] + (m + upper_bound) * num_of_vertices, id_dict[
                                            j] + m * num_of_vertices] = 1
                                elif Isdelay == False:
                                    A[id_dict[j] + m * num_of_vertices, id_dict[
                                        i] + m * num_of_vertices] = 1  # 选择T时刻(位于矩阵的左上角)作为起点 往右是T+1 往下是T+1
                            except IndexError:
                                continue
                    elif type_ == 'distance':
                        A[id_dict[i] + (day - upper_bound) * num_of_vertices, id_dict[j]] = 1 / distance

                    else:
                        raise ValueError("type_ error, must be "
                                         "connectivity or distance!")

        for i in range(day):  # 自身的循环
            for j in range(day):
                for k in range(len(id_dict)):
                    if i >= j:
                        A[k + i * num_of_vertices, k + j * num_of_vertices] = 1


    elif model == "stgcn":
        num_of_vertices = int(num_of_vertices)
        A = np.zeros([num_of_vertices] * 2)
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 4:
                    continue
                i, j, delay, distance = row[0], row[1], row[2], float(row[3])
                A[id_dict[i], id_dict[
                    j]] = 1 / (distance * 2) + 0.5 * (cor[id_dict[i], id_dict[j]])
                A[id_dict[j], id_dict[
                    i]] = 1 / (distance * 2) + 0.5 * (cor[id_dict[i], id_dict[j]])

        for a in range(num_of_vertices):
            A[a, a] = 1
    elif model == "gcn":
        num_of_vertices = int(num_of_vertices)
        A = np.zeros([num_of_vertices] * 2)
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 4:
                    continue
                i, j, delay, distance = row[0], row[1], row[2], float(row[3])
                if type_ == 'connectivity':
                    A[id_dict[i], id_dict[
                        j]] = 1
                    A[id_dict[j], id_dict[
                        i]] = 1
                else:
                    A[id_dict[i], id_dict[
                        j]] = 1 / (distance * 2)
                    A[id_dict[j], id_dict[
                        i]] = 1 / (distance * 2)
            for a in range(num_of_vertices):
                A[a, a] = 1
    return A


def construct_STSmodel(config, embedding_size,A,device):
    module_type = config['module_type']  # 模型类型
    act_type = config['act_type']  # GLU与Relu
    temporal_emb = config['temporal_emb']
    spatial_emb = config['spatial_emb']
    use_mask = config['use_mask']
    num_of_vertices = config['num_of_vertices']
    num_of_features = config['num_of_features']
    seq_input = config['seq_input']
    num_for_predict = config['num_for_predict']
    adj_filename = config['adj_filename']
    id_filename = config['id_filename']
    day = config["day"]
    adj_mx = torch.from_numpy(A).to(device)
    filters = config['filters']
    first_layer_embedding_size = embedding_size
    net = STSGCN(adj_mx, seq_input, num_of_vertices, num_of_features, num_for_predict, filters
                 , module_type, act_type, use_mask, temporal_emb, spatial_emb,
                 embedding_size=first_layer_embedding_size,window_size=day)
    return net.to(device)


def construct_alladj(id_filename, distance_df_filename, num_of_vertices, day=None,
                     type_='connectivity'):
    with open(id_filename, 'r') as f:
        id_dict = {i: idx
                   for idx, i in enumerate(f.read().strip().split('\n'))}
    import csv
    num_of_vertices = int(num_of_vertices)
    Adjs = []
    filter = set()
    A = np.zeros([num_of_vertices * day] * 2)
    N_info = []
    delay_info = []
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 4:
                continue
            i, j, delay, distance = row[0], row[1], row[2], float(row[3])
            if len(delay.split("-")) == 2:
                upper_bound, low_bound = delay.split("-")  # 滞后天数的上下界
                upper_bound = int(upper_bound)
                low_bound = int(low_bound) + 1 if low_bound is not '' else (upper_bound + 1)
                delay_info.append(list(range(upper_bound, low_bound)))
                N_info.append((id_dict[i], id_dict[j]))

    construct(len(id_dict),N_info, delay_info, day, np.zeros(len(N_info),dtype = np.int) , 0, Adjs,filter)
    # 地集点的大小、地点之间的关系、 延迟的界限、天数
    return Adjs

def construct(N, N_info, delay_info, day, indexes, cur, Adjs, filter):
    #for ix, l in enumerate(delay_info):
    #    i,j = N_info[ix]
    #    dy = l[indexes[ix]]
    #    for m in range(day):
   #         try:
    #            A[j + (m + dy) * N,
     #               i + m * N] = 1  # 选择T时刻(位于矩阵的左上角)作为起点 往右是T+1 往下是T+1
    #        except IndexError:
    #            continue

   # for ix in range(day):  # 自身的循环
    #    for k in range(N):
    #        A[k + ix * N, k] = 1
    #Adjs.append(A)]
    # cur  当前位置
    #id
    delay_all = []
    for id, u in enumerate(indexes):
        delay_all.append(delay_info[id][u])

    if tuple(delay_all) not in filter:
        filter.add(tuple(delay_all))
        A = np.zeros([N * day] * 2)
        for ix, l in enumerate(delay_info):
            i, j = N_info[ix]
            dy = min(l[indexes[ix]],day - 1)
            for m in range(day):
                 try:
                    A[j + (m + dy) * N,
                       i + m * N] = 1  # 选择T时刻(位于矩阵的左上角)作为起点 往右是T+1 往下是T+1
                 except IndexError:
                    continue
        for ix in range(day):  # 自身的循环
            for iy in range(day):
                for k in range(N):
                    if ix >= iy:
                        A[k + ix * N, k + iy * N] = 1
        Adjs.append((A,indexes))
    idxes = copy.copy(indexes)
    if indexes[cur] + 1 < len(delay_info[cur]):
        idxes[cur] = idxes[cur] + 1
        construct(N,N_info, delay_info, day, idxes, cur, Adjs, filter)
    if cur + 1 < len(N_info):
        construct(N,N_info, delay_info, day, indexes, cur + 1, Adjs, filter)

def generate_from_train_val_test(data, transformer):
    mean = None
    std = None
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(data[key], 12, 12)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y


def generate_from_data(data, length, transformer):
    mean = None
    std = None
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    for line1, line2 in ((0, train_line),
                         (train_line, val_line),
                         (val_line, length)):
        x, y = generate_seq(data['data'][line1: line2], 12, 12)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y


def generate_loaders(H5file, batch_size, transformer=None):
    """
        shape is (num_of_samples, day, num_of_vertices, 1)
    """
    with h5.File(H5file, 'r') as hf:
        train_X = hf["train_X"][:]
        train_Y = hf["train_Y"][:]
        validation_X = hf["validation_X"][:]
        validation_Y = hf["validation_Y"][:]
        test_X = hf["test_X"][:]
        test_Y = hf["test_Y"][:]
        cor = hf["cor"][:]
        cor_delay = hf["cor_delay"][:]
    X = np.concatenate([train_X, validation_X], axis=0)
    Y = np.concatenate([train_Y, validation_Y], axis=0)
    Y_copy = np.expand_dims(np.expand_dims(Y, axis=-1), axis=-1)
    show = np.concatenate([X, Y_copy], axis=-2)
    transformer.fit(show)
    train_X = transformer.transform(train_X)
    train_Y = transformer.transform(train_Y)
    validation_X = transformer.transform(validation_X)
    validation_Y = transformer.transform(validation_Y)
    test_X = transformer.transform(test_X)
    test_Y = transformer.transform(test_Y)
    datasets = [riverFlowDataset(stations=train_X, lake=train_Y),
                riverFlowDataset(stations=validation_X, lake=validation_Y),
                riverFlowDataset(stations=test_X, lake=test_Y)]
    loaders = []

    for dataset in datasets:
        loaders.append(DataLoader(dataset=dataset, batch_size=batch_size))
    loaders.append(cor)
    loaders.append(cor_delay)
    return loaders


def generate_data(graph_signal_matrix_filename, transformer=None):
    '''
    shape is (num_of_samples, 12, num_of_vertices, 1)
    '''
    data = np.load(graph_signal_matrix_filename)
    keys = data.keys()
    if 'train' in keys and 'val' in keys and 'test' in keys:
        for i in generate_from_train_val_test(data, transformer):
            yield i
    elif 'data' in keys:
        length = data['data'].shape[0]
        for i in generate_from_data(data, length, transformer):
            yield i
    else:
        raise KeyError("neither data nor train, val, test is in the data")


def generate_seq(data, train_length, pred_length):
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_length], 0)
        for i in range(data.shape[0] - train_length - pred_length + 1)],
        axis=0)[:, :, :, 0: 1]
    return np.split(seq, 2, axis=1)


def evaluate(predictions, targets, normal, T):
    """
    evaluate model with rmse
    :param predictions: [n_samples, T , n_nodes, n_features]
    :param targets: np.ndarray, shape [n_samples, length , n_nodes, n_features]
    :param T:time step
    :param normal:contain mean and std
    :return: dict
    """
    n_samples = targets.shape[0]
    scores = defaultdict(dict)
    for horizon in range(T):
        y_true = np.reshape(targets[:, horizon], (n_samples, -1))
        y_pred = np.reshape(predictions[:, horizon], (n_samples, -1))
        scores['MAE'][f'horizon-{horizon + 1}'] = normal.mae_transform(mean_absolute_error(y_pred, y_true))
        scores['RMSE'][f'horizon-{horizon + 1}'] = normal.rmse_transform(np.sqrt(mean_squared_error(y_pred, y_true)))
        # scores['MAPE'][f'horizon-{horizon}'] = masked_mape_np(y_pred, y_true) * 100.0
    y_true = np.reshape(targets, (n_samples, -1))
    y_pred = np.reshape(predictions, (n_samples, -1))
    scores['rmse'] = normal.rmse_transform(np.sqrt(mean_squared_error(y_true, y_pred)))
    scores['mae'] = normal.mae_transform(mean_absolute_error(y_pred, y_true))
    return scores


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def showDelayTime(id_filename, distance_df_filename, delay_time):
    res = {}
    with open(id_filename, 'r') as f:
        id_dict = {i: idx
                   for idx, i in enumerate(f.read().strip().split('\n'))}

        id_dict_new = dict(zip(id_dict.values(), id_dict.keys()))
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        N_info = []
        delay_info = []
        for row in reader:
            if len(row) != 4:
                continue
            i, j, delay, distance = row[0], row[1], row[2], float(row[3])
            if len(delay.split("-")) == 2:
                upper_bound, low_bound = delay.split("-")  # 滞后天数的上下界
                upper_bound = int(upper_bound)
                low_bound = int(low_bound) + 1 if low_bound is not '' else (upper_bound + 1)
                N_info.append((id_dict[i], id_dict[j]))
                delay_info.append(list(range(upper_bound, low_bound)))
        for idx,con in enumerate(N_info):
            print("{0}->{1}:{2}".format(id_dict_new[con[0]],id_dict_new[con[1]],delay_info[idx][delay_time[idx]]))
