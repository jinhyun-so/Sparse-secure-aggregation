#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
from sklearn import metrics


def average_weights(w, is_coef=False, coef=None, device_id=-1):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    if device_id > -1:
        device = torch.device('cuda:{}'.format(device_id))
    else:
        device = None

    if is_coef:
        for key in w_avg.keys():
            denom = 0.0
            if device_id > -1:
                w_avg[key] = w_avg[key].to(device)
            else:
                w_avg[key] = w_avg[key].detach().cpu()

            for i in range(1, len(w)):
                if device_id > -1:
                    tmp = w[i][key].to(device)
                    w_avg[key] += tmp * coef[i]
                else:
                    tmp = w[i][key].detach().cpu()
                    w_avg[key] += tmp * coef[i]                
                denom += coef[i]
            w_avg[key] = torch.div(w_avg[key], len(w))
    else:
        for key in w_avg.keys():
            if device_id > -1:
                w_avg[key] = w_avg[key].to(device)
            else:
                w_avg[key] = w_avg[key].detach().cpu()

            for i in range(1, len(w)):
                if device_id > -1:
                    tmp = w[i][key].to(device)
                    w_avg[key] += tmp
                else:
                    tmp = w[i][key].detach().cpu()
                    w_avg[key] += tmp
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def add_weights(w1, w2, device_id=-1):
    """
    Returns the average of the weights.
    """
    w0 = copy.deepcopy(w1)
    for key in w0.keys():
        # print(key, type(w0[key]))
        # print(key, type(w2[key]))
        # print(key)
        if device_id == -1:
            w0[key] = w0[key].float().detach().cpu()
            w0[key] += w2[key].float().detach().cpu()
        else:
            device = torch.device('cuda:{}'.format(device_id))
            w0[key] = w0[key].float().to(device).float()
            w0[key] += w2[key].to(device).float()
    return w0

def sub_weights(w1, w2, device_id=-1):
    """
    Returns the average of the weights.
    """
    w0 = copy.deepcopy(w1)
    for key in w0.keys():
        if device_id == -1:
            w0[key] = w0[key].float().detach().cpu()
            w0[key] -= w2[key].float().detach().cpu()
        else:
            w0[key] = w0[key].float()
            w0[key] -= w2[key].float()
    return w0

def scale_weights(w1, scale):
    for key in w1.keys():
        w1[key] = w1[key].float() * scale
    return w1

def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)
    
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, is_send_gradient = False):
        
        if is_send_gradient == True:
            prev_net = copy.deepcopy(net)
        net.train()
        # train and update
        if self.args.opt == 'ADAM':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=5e-4)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = self.args.weight_decay)
        #optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
        #optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        #optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=5e-4)

        #print('train starts here!')
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                '''
                if self.args.verbose and batch_idx % 10 == 0: #self.args.verbose
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                '''
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        if is_send_gradient == True:
            return sub_weights(net.state_dict(), prev_net.state_dict()), sum(epoch_loss) / len(epoch_loss)
        else:
            return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
        

