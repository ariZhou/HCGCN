# -*- coding:utf-8 -*-
import copy
import math
import os
import json
import argparse

import numpy as np
from tqdm import tqdm
import torch as th
from utils import (construct_model, generate_data, generate_loaders,construct_alladj,construct_STSmodel,showDelayTime,
                   masked_mae_np, masked_mape_np, masked_mse_np, evaluate, seed_everything, MyEncoder)
from normalization import Standard
from collections import defaultdict
from data_container import RMSELoss
import heapq

class Result:

    def __init__(self,score, adj, emb):
        self.score = score
        self.adj = adj
        self.emb = emb

    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return math.isclose(self.score, other.score, rel_tol= 0.000001)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return "Adj:{0},Emb:{1},score:{2}".format(self.adj, self.emb, self.score)

    def __repr__(self):
        return "Adj:{0},Emb:{1},score:{2}".format(self.adj, self.emb, self.score)
th.set_default_tensor_type(th.DoubleTensor)
parser = argparse.ArgumentParser()
parser.add_argument("--mode",type=str,choices=["run","check","show","find"],default="run")
parser.add_argument("--adj",type=int,help='matrix index')
parser.add_argument("--emb",type=int,help='embbeding size')
parser.add_argument("--config", type=str, help='configuration file')
args = parser.parse_args()
config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
device = th.device("cuda" if th.cuda.is_available() else "cpu",0)
batch_size = config['batch_size']
H5matrix_filename = config['H5FileName']
seq_input = config["seq_input"]
early_stop_steps = config["early_stop_steps"]
true_values = []
normal = Standard()
"""
cor 和cor_delay不是loader
"""


folder = os.path.join('save',"stsMat")
best_des = os.path.join(folder, "best.pkl")
if args.mode == "run":
    Adjs = construct_alladj(config['id_filename'], config['adj_filename'],config["num_of_vertices"],day=config["day"])
    rmse = RMSELoss()
    max_grad_norm = 5
    all_info = []
    epochs = config['epochs']
    folder = os.path.join('save',"stsMat")
    begin_epoch = 0
    begin_mat = 0
    best_emb = 0
    best_test_val = float('inf')
    if os.path.exists(best_des):
        best_dict = th.load(best_des)
        begin_mat = best_dict["begin_mat"]
    else:
        best_dict = dict(best_test_val=float('inf'))
    for idx in range(0, len(Adjs)):
        print("select {0} adj".format(idx))
        best_dict.update(begin_mat=idx)
        th.save(best_dict, os.path.join(folder, "best.pkl"))
        emb_dest = os.path.join(folder, str(idx))
        emb_dict = dict(emb_test_val=float('inf'))
        for emb in range(1,51):
            print("Start {0} embbeding size".format(emb))
            seed_everything(2022)
            save_dict = dict()
            os.makedirs(emb_dest, exist_ok=True)
            pkl_dest = os.path.join(emb_dest, str(emb))
            os.makedirs(pkl_dest, exist_ok=True)
            save_path = os.path.join(pkl_dest, 'best_model.pkl')
            adj = Adjs[idx][0]
            net = construct_STSmodel(config, emb, adj, device)
            optimizer = th.optim.Adam(net.parameters(), 1e-3)
            scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=4, threshold=0.001,
                                                            min_lr=0.000001)
            train_loader, validate_loader, test_loader, cor, cor_delay = generate_loaders(H5matrix_filename, batch_size, normal)
            phases = [("train", train_loader), ("validate", validate_loader)]
            best_val_loss = float('inf')
            for epoch in range(begin_epoch, begin_epoch + epochs):
                needBreak = False
                running_loss, running_metrics = defaultdict(float), dict()
                for phase, loader in phases:
                    if phase == "train":
                        net.train()
                    else:
                        net.eval()
                    steps, predictions, running_targets = 0, list(), list()
                    tqdm_loader = enumerate(loader)
                    for step, (inputs, targets) in tqdm_loader:
                        '''
                            针对（B，T）格式的硬编码
                        '''
                        targets = th.unsqueeze(th.unsqueeze(targets, dim=-1), dim=-1)
                        '''
            
                        '''
                        running_targets.append(targets.numpy())
                        with th.set_grad_enabled(phase == 'train'):
                            inputs_g = inputs.to(device)
                            targets_g = targets.to(device)
                            outputs = net(inputs_g)
                            loss = rmse(targets_g, outputs)
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
                        #tqdm_loader.set_description(
                        #   f'{phase:5} epoch: {epoch:3}. {phase:5} loss: {normal.rmse_transform(running_loss[phase] / steps):3.6}')
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
                            #print(f'Better model at epoch {epoch} recorded.')
                        elif epoch - save_dict['epoch'] > early_stop_steps or epoch == begin_epoch + epochs - 1:
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
                                inputs_tg = inputs_t.to(device)
                                with th.no_grad():
                                    outputs_t = net(inputs_tg)
                                predictions.append(outputs_t.cpu().numpy())
                            running_targets, predictions = np.concatenate(running_targets, axis=0), \
                                                           np.concatenate(predictions, axis=0)
                            scores = evaluate(running_targets, predictions, normal, config["num_for_predict"])
                            #print('test results:')
                            #print(json.dumps(scores, cls=MyEncoder, indent=4))
                            with open(os.path.join(pkl_dest, 'test-scores.json'), 'w+') as f:
                                json.dump(scores, f, cls=MyEncoder, indent=4)
                            #with open(os.path.join(dest, 'config.json'), 'w') as cf:
                            #    json.dump(config, cf, cls=MyEncoder, indent=4)
                            if scores['mae'] <= emb_dict.get("emb_test_val"):
                                emb_dict.update(emb_test_val=scores['mae'], emb_size=emb)

                            if scores['mae'] <= best_dict.get("best_test_val"):
                                best_dict.update(best_test_val=scores['mae'],best_emb_size=emb,best_adj_idx=idx)
                                th.save(best_dict, os.path.join(folder, "best.pkl"))
                                print(f'best emb , score and  adj  recorded.')
                            needBreak = True
                    scheduler.step(running_loss['train'])
                if needBreak:
                    break
        th.save(emb_dict, os.path.join(emb_dest, "emb.pkl"))
        print(f'best emb for  adj at {idx} recorded.')

elif args.mode == "check":
    adj = args.adj
    emb = args.emb
    if adj is None and emb is None:
        print("adj and emb are alL None")
        best_dict = th.load(os.path.join(folder,"best.pkl"))
        print("MAE:{0},Adj_idx:{1},Embedding:{2}".format(best_dict["best_test_val"],best_dict["best_adj_idx"],best_dict["best_emb_size"]))
    elif emb is None:
        print("emb is None")
        dest = os.path.join(folder, str(adj))
        emb_dict = th.load(os.path.join(dest, "emb.pkl"))
        print("MAE:{0},Embedding:{1}".format(emb_dict["best_test_val'"], emb_dict["emb_size"]))

elif args.mode == "show":
    Adjs = construct_alladj(config['id_filename'], config['adj_filename'], config["num_of_vertices"], day=config["day"])
    showDelayTime(config['id_filename'],config['adj_filename'],Adjs[93][1])
elif args.mode == "find":
    des = "save/stsMat"
    heap = []
    for adj in os.listdir(des):
        if adj.endswith(".pkl"):
            continue
        for emb in os.listdir(os.path.join(des,adj)):
            if emb.endswith(".pkl") or os.path.exists(os.path.join(des, adj, emb,"test-scores.json")) == False:
                continue
            with open(os.path.join(des, adj, emb,"test-scores.json")) as jf:
                j = json.load(jf)
            r = Result(j["mae"], int(adj), int(emb))
            heap.append(r)

    print(heapq.nsmallest(100,heap).__str__())
