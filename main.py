# -*- coding:utf-8 -*-
import copy
import math
import os
import time
import json
import argparse

import numpy as np
from tqdm import tqdm
import torch as th
from utils import (construct_model, generate_data, generate_loaders,
                   masked_mae_np, masked_mape_np, masked_mse_np, evaluate, seed_everything, MyEncoder)
from normalization import Standard
from collections import defaultdict
from data_container import RMSELoss

th.set_default_tensor_type(th.DoubleTensor)
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--test", action="store_true", help="test program")
parser.add_argument("--plot", help="plot network graph", action="store_true")
parser.add_argument("--save", action="store_true", help="save model")
args = parser.parse_args()
config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())

print(json.dumps(config, sort_keys=True, indent=4))
seed_everything(2022)

batch_size = config['batch_size']
H5matrix_filename = config['H5FileName']
seq_input = config["seq_input"]
early_stop_steps = config["early_stop_steps"]

true_values = []
normal = Standard()
"""
cor 和cor_delay不是loader
"""
train_loader, validate_loader, test_loader, cor, cor_delay = generate_loaders(H5matrix_filename, batch_size, normal)
config["cor"] = cor
config["cor_delay"] = cor_delay
phases = [("train", train_loader), ("validate", validate_loader)]

net = construct_model(config)
del config["cor"]
del config["cor_delay"]
optimizer = th.optim.Adam(net.parameters(), 1e-3)
scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=4, threshold=0.001, min_lr=0.000001)
rmse = RMSELoss()
max_grad_norm = 5

val_value = []  # 验证集结果
test_value = []  # 测试集结果

global_epoch = 1
global_train_steps = len(train_loader) // batch_size + 1
all_info = []
epochs = config['epochs']
folder = os.path.join('save1', config['name'])
os.makedirs(folder, exist_ok=True)
save_path = os.path.join(folder, 'best_model.pkl')



#if os.path.exists(save_path):
#    print("path exist")
#    save_dict = th.load(save_path)

#    net.load_state_dict(save_dict['model_state_dict'])
#    optimizer.load_state_dict(save_dict['optimizer_state_dict'])

 #   best_val_loss = save_dict['best_val_loss']
#    begin_epoch = save_dict['epoch'] + 1
#else:
   # print("path does not exist")
save_dict = dict()
best_val_loss = float('inf')
begin_epoch = 0

for epoch in range(begin_epoch, begin_epoch + epochs):
    running_loss, running_metrics = defaultdict(float), dict()
    for phase, loader in phases:
        if phase == "train":
            net.train()
        else:
            net.eval()
        steps, predictions, running_targets = 0, list(), list()
        tqdm_loader = tqdm(enumerate(loader))
        for step, (inputs, targets) in tqdm_loader:
            '''
                针对（B，T）格式的硬编码
            '''
            targets = th.unsqueeze(th.unsqueeze(targets, dim=-1), dim=-1)
            '''

            '''
            running_targets.append(targets.numpy())
            with th.set_grad_enabled(phase == 'train'):
                outputs = net(inputs)
                loss = rmse(targets, outputs)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    if max_grad_norm is not None:
                        th.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                    optimizer.step()
            with th.no_grad():
                predictions.append(outputs.cpu().numpy())
            running_loss[phase] += ((loss ** 2) * len(targets))
            steps += len(targets)
            tqdm_loader.set_description(
                f'{phase:5} epoch: {epoch:3}. {phase:5} loss: {normal.rmse_transform(running_loss[phase] / steps):3.6}')
            th.cuda.empty_cache()
        running_metrics[phase] = evaluate(np.concatenate(predictions), np.concatenate(running_targets), normal,
                                          config["num_for_predict"])

        if phase == 'validate':
            if running_loss['validate'] <= best_val_loss or math.isnan(running_loss['validate']):
                best_val_loss = running_loss['validate']
                save_dict.update(model_state_dict=copy.deepcopy(net.state_dict()),
                                 epoch=epoch,
                                 best_val_loss=best_val_loss,
                                 optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                th.save(save_dict, save_path)
                print(f'Better model at epoch {epoch} recorded.')
            elif epoch - save_dict['epoch'] > early_stop_steps:
                net.load_state_dict(save_dict['model_state_dict'])
                net.eval()
                steps, predictions, running_targets = 0, list(), list()
                tqdm_loader = tqdm(enumerate(test_loader))
                for step, (inputs_t, targets_t) in tqdm_loader:
                    '''
                         针对（B，T）格式的硬编码
                    '''
                    targets_t = th.unsqueeze(th.unsqueeze(targets_t, dim=-1), dim=-1)
                    '''
s
                    '''
                    running_targets.append(targets_t.numpy())
                    with th.no_grad():
                        outputs_t = net(inputs_t)
                    predictions.append(outputs_t.cpu().numpy())
                running_targets, predictions = np.concatenate(running_targets, axis=0),\
                                               np.concatenate(predictions, axis=0)

                scores = evaluate(running_targets[:,:config["num_for_predict"] - 2], predictions[:,:config["num_for_predict"] - 2], normal, config["num_for_predict"] - 2)
                print('test results:')
                print(json.dumps(scores, cls=MyEncoder, indent=4))
                with open(os.path.join(folder, 'test-scores.json'), 'w+') as f:
                    json.dump(scores, f, cls=MyEncoder, indent=4)

                with open(os.path.join(folder,'config.json'),'w') as cf:
                    json.dump(config,cf,cls=MyEncoder,indent=4)
                raise ValueError('Early stopped.')
        scheduler.step(running_loss['train'])


net.load_state_dict(save_dict['model_state_dict'])
net.eval()
steps, predictions, running_targets = 0, list(), list()
tqdm_loader = tqdm(enumerate(test_loader))
for step, (inputs_t, targets_t) in tqdm_loader:
    '''
         针对（B，T）格式的硬编码
    '''
    targets_t = th.unsqueeze(th.unsqueeze(targets_t, dim=-1), dim=-1)
    '''

    '''
    running_targets.append(targets_t.numpy())
    with th.no_grad():
        outputs_t = net(inputs_t)
    predictions.append(outputs_t.cpu().numpy())
running_targets, predictions = np.concatenate(running_targets, axis=0),\
                               np.concatenate(predictions, axis=0)
scores = evaluate(running_targets[:, :config["num_for_predict"] - 2], predictions[:, :config["num_for_predict"] - 2],
                  normal, config["num_for_predict"] - 2)
print('test results:')
print(json.dumps(scores, cls=MyEncoder, indent=4))
with open(os.path.join(folder, 'test-scores.json'), 'w+') as f:
    json.dump(scores, f, cls=MyEncoder, indent=4)
with open(os.path.join(folder,'config.json'),'w') as cf:
    json.dump(config,cf,cls=MyEncoder,indent=4)