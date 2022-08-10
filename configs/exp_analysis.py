import os
import os.path as osp
import copy
import logging
import numpy as np
import pandas as pd
import torch

from configs.config_global import ROOT_DIR, LOG_LEVEL
from utils.config_utils import configs_dict_unpack
from analysis import plots
from configs import experiments

def get_curve(cfgs, idx, key='normalized_reward', max_batch=None, num_points=20, start=0):

    num_seeds = len(cfgs)
    
    if max_batch is None:
        max_batch = cfgs[0][idx].max_batch
    
    eval_interval = cfgs[0][idx].log_every
    tot_steps = max_batch // eval_interval
    plot_every = tot_steps // num_points
    performance = []

    for seed in range(num_seeds):
        cfg = cfgs[seed][idx]

        try:
            file_path = osp.join(cfg.save_path, 'progress.txt')
            exp_data = pd.read_table(file_path)
            acc = exp_data[key][start: tot_steps + 1: plot_every]
            
            if len(acc) >= (tot_steps - start) // plot_every + 1:
                performance.append(acc)
            else:
                raise ValueError

        except:
            print("Incomplete data:", cfg.save_path)

    performance = np.stack(performance, axis=1)
    x_axis = np.arange(start * eval_interval, max_batch + 1, plot_every * eval_interval)

    return performance, x_axis

def get_performance(cfgs, idx, key1='TrainLoss', key2='TestError', check_complete=True, file_name='progress.txt', beg=1):

    num_seeds = len(cfgs)
    performance = []

    max_batch = cfgs[0][idx].max_batch
    eval_interval = cfgs[0][idx].log_every
    tot_steps = max_batch // eval_interval + 1

    for seed in range(num_seeds):
        cfg = cfgs[seed][idx]

        try:
            file_path = osp.join(cfg.save_path, file_name)
            exp_data = pd.read_table(file_path)

            if check_complete and len(exp_data[key1]) < tot_steps:
                raise ValueError(exp_data)

            best_t = exp_data[key1][beg: ].argmin()
            performance.append(exp_data[key2][best_t + beg])

        except:
            print("Incomplete data:", cfg.save_path)

    return performance

def get_test_mean_ci(cfgs, idx):
    performance = get_performance(cfgs, idx, key1='TestLoss', key2='TestAcc', file_name='test.txt', beg=0, check_complete=False)
    return np.mean(performance), plots.get_sem(performance) * 1.96

def seqreproduction_long_compare_delay_analysis():
    cfgs = experiments.seqreproduction_long_compare_delay()
    model_list = ['RNN_Gradient', 'RNN_Non-Plastic', 'LSTM_Gradient', 'LSTM_Non-Plastic']

    for j, delay in enumerate([0, 10, 20, 40]):

        performance = []

        for i, model in enumerate(model_list):
            curve, x_axis = get_curve(cfgs, i + j * 4, key='TestLoss', start=1)
            performance.append(curve)

        plots.error_range_plot(
            x_axis,
            performance,
            x_label='Training Step',
            y_label='Validation Loss',
            label_list=model_list,
            fig_dir='seq',
            fig_name=f'long_delay{delay}'
        )

def seqreproduction_short_compare_delay_analysis():
    cfgs = experiments.seqreproduction_short_compare_delay()
    model_list = ['RNN_Gradient', 'RNN_Non-Plastic', 'LSTM_Gradient', 'LSTM_Non-Plastic']

    for j, delay in enumerate([10, 20, 30]):

        performance = []

        for i, model in enumerate(model_list):
            curve, x_axis = get_curve(cfgs, i + j * 4, key='TestLoss', start=1)
            performance.append(curve)

        plots.error_range_plot(
            x_axis,
            performance,
            x_label='Training Step',
            y_label='Validation Loss',
            label_list=model_list,
            fig_dir='seq',
            fig_name=f'short_delay{delay}'
        )

def cuereward_analysis():
    
    cfgs = experiments.cuereward()
    model_list = ['RNN_Gradient', 'RNN_Non-Plastic', 'LSTM_Gradient', 'LSTM_Non-Plastic']
    performance = []

    for i, model in enumerate(model_list):

        curve, x_axis = get_curve(cfgs, i, key='TestLoss', start=1)
        performance.append(curve)

    plots.error_range_plot(
        x_axis,
        performance,
        x_label='Training Step',
        y_label='Validation Loss',
        label_list=model_list,
        fig_dir='cuereward',
        fig_name=f'compare_plasticity'
    )

def cuereward_large_analysis():
    
    cfgs = experiments.cuereward_large()
    model_list = ['RNN_Gradient', 'RNN_Non-Plastic', 'LSTM_Gradient', 'LSTM_Non-Plastic']
    performance = []

    for i, model in enumerate(model_list):

        curve, x_axis = get_curve(cfgs, i, key='TestLoss', start=1)
        performance.append(curve)

    plots.error_range_plot(
        x_axis,
        performance,
        x_label='Training Step',
        y_label='Validation Loss',
        label_list=model_list,
        fig_dir='cuereward',
        fig_name=f'compare_plasticity_large'
    )

def cuereward_inner_lr_mode_analysis():

    cfgs = experiments.cuereward_inner_lr_mode()
    model_list = ['none', 'uniform', 'random']

    for j, rnn in enumerate(['RNN', 'LSTM']):
        performance = []

        for i, model in enumerate(model_list):

            curve, x_axis = get_curve(cfgs, i + j * 3, key='TestLoss', start=1)
            performance.append(curve)

        plots.error_range_plot(
            x_axis,
            performance,
            x_label='Training Step',
            y_label='Validation Loss',
            label_list=model_list,
            fig_dir='cuereward',
            fig_name=f'inner_lr_{rnn}'
        )

def cuereward_lr_analysis():
    cfgs = experiments.cuereward_lr()

    for idx, lr in enumerate([0.01, 0.001, 0.0003]):
        model_list = ['RNN_Gradient', 'RNN_Non-Plastic', 'LSTM_Gradient', 'LSTM_Non-Plastic']
        performance = []

        for i, model in enumerate(model_list):

            curve, x_axis = get_curve(cfgs, idx * 4 + i, key='TestLoss', start=1)
            performance.append(curve)

        plots.error_range_plot(
            x_axis,
            performance,
            x_label='Training Step',
            y_label='Validation Loss',
            label_list=model_list,
            fig_dir='cuereward',
            fig_name=f'compare_plasticity_lr{lr}'
        )

def cuereward_extradim_analysis():
    cfgs = experiments.cuereward_extradim()

    models = ['RNN', 'LSTM', ]
    performances = []
    x_axis = [0, 4, 8, 16]

    for i, model in enumerate(models):

        performances.append([])
        for idx, d in enumerate(x_axis):
            performances[-1].append(get_performance(cfgs, i * 4 + idx))

    plots.errorbar_plot(
        x_axis,
        performances,
        'Dim',
        'Validation Error',
        models,
        fig_dir='cuereward',
        fig_name='extradim'
    )

def cuereward_plr_analysis():
    cfgs = experiments.cuereward_plr()

    models = ['RNN', 'LSTM', ]
    performances = []
    x_axis = [0.02, 0.05, 0.1, 0.2]

    for i, model in enumerate(models):

        performances.append([])
        for idx, d in enumerate(x_axis):
            performances[-1].append(get_performance(cfgs, i * 4 + idx))

    plots.errorbar_plot(
        x_axis,
        performances,
        'Inner lr',
        'Validation Error',
        models,
        fig_dir='cuereward',
        fig_name='innerlr'
    )

def fsc_analysis():
    cfgs = experiments.fsc()

    dataset_list = ['CIFAR_FS', 'miniImageNet']
    for j, dataset in enumerate(dataset_list):

        model_list = ['LSTM_Gradient', 'RNN_Gradient', 'LSTM_Non-Plastic', 'RNN_Non-Plastic', ]
        performance = []

        print(dataset)

        for i, model in enumerate(model_list):
            performance.append(get_test_mean_ci(cfgs, i + j * 4))
            mean, ci = performance[-1]
            print(model, ":", mean, ci)

def fsc_resnet_analysis():
    cfgs = experiments.fsc_resnet()

    dataset_list = ['CIFAR_FS', 'miniImageNet']
    for j, dataset in enumerate(dataset_list):

        model_list = ['LSTM_Gradient', 'RNN_Gradient', 'LSTM_Non-Plastic', 'RNN_Non-Plastic', ]
        performance = []

        print(dataset)

        for i, model in enumerate(model_list):
            performance.append(get_test_mean_ci(cfgs, i + j * 4))
            mean, ci = performance[-1]
            print(model, ":", mean, ci)

def fsc_simclr_pretrain_analysis():
    cfgs = experiments.fsc_simclr_pretrain()

    dataset_list = ['miniImageNet_res', 'miniImageNet', 'CIFAR_FS_res', 'CIFAR_FS', ]
    for j, dataset in enumerate(dataset_list):

        model_list = ['LSTM_Gradient', 'RNN_Gradient', 'LSTM_Non-Plastic', 'RNN_Non-Plastic', ]
        performance = []

        print(dataset)

        for i, model in enumerate(model_list):
            performance.append(get_test_mean_ci(cfgs, i + j * 4))
            mean, ci = performance[-1]
            print(model, ":", mean, ci)

def fsc_lr_analysis():
    cfgs = experiments.fsc_lr()

    dataset_list = [0.01, 0.003, 0.0003]
    for j, dataset in enumerate(dataset_list):

        model_list = ['LSTM_Gradient', 'RNN_Gradient', 'LSTM_Non-Plastic', 'RNN_Non-Plastic', ]
        performance = []

        print(dataset)

        for i, model in enumerate(model_list):
            performance.append(get_test_mean_ci(cfgs, i + j * 4))
            mean, ci = performance[-1]
            print(model, ":", mean, ci)

def fsc_wd_analysis():
    cfgs = experiments.fsc_wd()

    dataset_list = [0.01, 0.002, 0.0001]
    for j, dataset in enumerate(dataset_list):

        model_list = ['LSTM_Gradient', 'RNN_Gradient', 'LSTM_Non-Plastic', 'RNN_Non-Plastic', ]
        performance = []

        print(dataset)

        for i, model in enumerate(model_list):
            performance.append(get_test_mean_ci(cfgs, i + j * 4))
            mean, ci = performance[-1]
            print(model, ":", mean, ci)

def fsc_plr_analysis():
    cfgs = experiments.fsc_plr()

    dataset_list = [0, 0.03, 0.1, 0.2]
    for j, dataset in enumerate(dataset_list):

        model_list = ['LSTM_Gradient', 'RNN_Gradient', ]
        performance = []

        print(dataset)

        for i, model in enumerate(model_list):
            performance.append(get_test_mean_ci(cfgs, i + j * 2))
            mean, ci = performance[-1]
            print(model, ":", mean, ci)

def fsc_extradim_analysis():
    cfgs = experiments.fsc_extradim()

    dataset_list = [0, 4, 16, 64]
    for j, dataset in enumerate(dataset_list):

        model_list = ['LSTM_Gradient', 'RNN_Gradient', ]
        performance = []

        print(dataset)

        for i, model in enumerate(model_list):
            performance.append(get_test_mean_ci(cfgs, i + j * 2))
            mean, ci = performance[-1]
            print(model, ":", mean, ci)

def fsc_pretrain_analysis():
    cfgs = experiments.fsc_pretrain()

    dataset_list = ['CIFAR_FS', 'miniImageNet']
    for j, dataset in enumerate(dataset_list):

        model_list = ['LSTM_Gradient', 'RNN_Gradient', 'LSTM_Non-Plastic', 'RNN_Non-Plastic', ]
        performance = []

        print(dataset)

        for i, model in enumerate(model_list):
            performance.append(get_test_mean_ci(cfgs, i + j * 4))
            mean, ci = performance[-1]
            print(model, ":", mean, ci)