from copy import deepcopy
import torch
import os.path as osp
import torch.nn as nn
import models

from configs.config_global import ROOT_DIR, DEVICE, MAP_LOC
from collections import OrderedDict

def get_cnn(cnn_type, config=None):

    if config == None or config.cnn_pretrain == 'none':

        if cnn_type == 'ProtoNet':
            network = models.ProtoNetEmbedding(3, 64, 64)
        elif cnn_type == 'R2D2':
            network = models.R2D2Embedding()
        elif cnn_type == 'ResNet':
            if config is not None and config.input_shape[1] == 84: # For MiniImageNet
                network = models.resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5)
                # network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
            else:
                network = models.resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2)
        else:
            raise NotImplementedError("Cannot recognize the network type")

    elif config.cnn_pretrain in ['classification', 'contrastive']:
        cnn_path =  f"experiments/{config.cnn_pretrain}_pretrain/" + \
                    f"model_image_dataset{config.image_dataset}_cnn{config.cnn}_s{config.seed}" + \
                    f"/cnn_{config.pretrain_step}.pth"
        
        network = torch.load(cnn_path)
        if config.freeze_pretrained_cnn:
            network.requires_grad_(False)

    else:
        raise NotImplementedError("Pretrain method not implemented")

    network.eval()
    network = network.to('cpu')
    return network

def get_rnn(rnn_type, plastic_mode, rnn_in_size, hidden_size):
    if rnn_type == 'RNN' and plastic_mode == 'none':
        rnn = nn.RNNCell(rnn_in_size, hidden_size, nonlinearity='relu')
    elif rnn_type == 'LSTM' and plastic_mode == 'none':
        rnn = nn.LSTMCell(rnn_in_size, hidden_size)
    elif rnn_type == 'RNN' and plastic_mode == 'gradient':
        rnn = models.PlasticRNNCell(rnn_in_size, hidden_size)
    elif rnn_type == 'LSTM' and plastic_mode == 'gradient':
        rnn = models.PlasticLSTMCell(rnn_in_size, hidden_size)
    else:
        raise NotImplementedError('RNN not implemented')

    return rnn

def get_linear(plastic_mode, in_size, out_size):
    if plastic_mode == 'none':
        layer = models.Linear(in_size, out_size)
    elif plastic_mode == 'gradient':
        layer = models.PlasticLinear(in_size, out_size)
    else:
        raise NotImplementedError('Layer not implemented')

    return layer