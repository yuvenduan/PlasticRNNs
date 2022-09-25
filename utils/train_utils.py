import os.path as osp

import torch
from torch.nn.utils import clip_grad_norm_
from configs.configs import BaseConfig
import models.model as models
from tasks import taskfunctions
from configs.config_global import ROOT_DIR, DEVICE, MAP_LOC

def grad_clipping(model, max_norm, printing=False):
    p_req_grad = [p for p in model.parameters() if p.requires_grad]

    if printing:
        grad_before = 0.0
        for p in p_req_grad:
            param_norm = p.grad.data.norm(2)
            grad_before += param_norm.item() ** 2
        grad_before = grad_before ** (1. / 2)

    clip_grad_norm_(p_req_grad, max_norm)

    if printing:
        grad_after = 0.0
        for p in p_req_grad:
            param_norm = p.grad.data.norm(2)
            grad_after += param_norm.item() ** 2
        grad_after = grad_after ** (1. / 2)

        if grad_before > grad_after:
            print("clipped")
            print("before: ", grad_before)
            print("after: ", grad_after)


def model_init(config_: BaseConfig, mode, net_num=None):
    if net_num is not None:
        assert mode == 'eval', 'net number only provided at eval mode'

    if config_.model_type == 'SimpleRNN':
        model = models.SimpleRNN(config_)
    elif config_.model_type == 'CNNtoRNN':
        model = models.CNNtoRNN(config_)
    elif config_.model_type == 'CNN':
        model = models.CNN(config_)
    elif config_.model_type == 'RecurrentPolicy':
        model = models.RecurrentPolicy(config_)
    elif config_.model_type == 'ActorCritic':
        model = models.ActorCritic(config_)
    else:
        raise NotImplementedError("Model not Implemented")

    if config_.load_path is not None:
        model.load_state_dict(torch.load(config_.load_path))
        
    model.to(DEVICE)
    return model

def task_init(config_: BaseConfig):
    """initialize tasks"""
    task_type = config_.task_type

    if task_type == 'SeqRegression':
        task_func_ = taskfunctions.SeqRegression(config_)
    elif task_type == 'FSC':
        task_func_ = taskfunctions.FSC(config_)
    elif task_type == 'Classification':
        task_func_ = taskfunctions.Classification(config_)
    elif task_type == 'Contrastive':
        task_func_ = taskfunctions.ContrastiveLearning(config_)
    else:
        raise NotImplementedError('task not implemented')

    return task_func_
