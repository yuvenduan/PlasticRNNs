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

def all_analysis():

    cuereward_large_analysis()
    cuereward_extradim_analysis()
    cuereward_inner_lr_mode_analysis()
    cuereward_lr_curves_analysis()
    cuereward_modulation_analysis()
    cuereward_plr_analysis()
    cuereward_lr_analysis()

    fsc_resnet_curves_analysis()
    fsc_curves_analysis()

    seqreproduction_delay_analysis()
    seqreproduction_seqlength_analysis()
    seqreproduction_long_compare_delay_analysis()
    seqreproduction_lr_curves_analysis()
    seqreproduction_modulation_analysis()

    regression_mlp_curves_analysis()
    regression_mlp_modulation_analysis()

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

def get_performance(
    cfgs, idx, 
    key1='TrainLoss', key2='TestError', 
    check_complete=True, file_name='progress.txt', 
    beg=1, cfgs2=None
):

    num_seeds = len(cfgs)
    performance = []

    max_batch = cfgs[0][idx].max_batch
    eval_interval = cfgs[0][idx].log_every
    tot_steps = max_batch // eval_interval + 1

    for seed in range(num_seeds):
        cfg = cfgs[seed][idx]

        if cfgs2 is not None:
            cfg2 = cfgs2[seed][idx]

        try:
            file_path = osp.join(cfg.save_path, file_name)
            exp_data = pd.read_table(file_path)

            if check_complete and len(exp_data[key1]) < tot_steps:
                raise ValueError(exp_data)

            if cfgs2 is not None:
                max_batch2 = cfgs2[0][idx].max_batch
                eval_interval2 = cfgs2[0][idx].log_every
                tot_steps2 = max_batch2 // eval_interval2 + 1
                file_path2 = osp.join(cfg2.save_path, 'progress.txt')
                exp_data2 = pd.read_table(file_path2)

                if len(exp_data2[key1]) < tot_steps2:
                    raise ValueError(exp_data)

            best_t = exp_data[key1][beg: ].argmin()
            performance.append(exp_data[key2][best_t + beg])

        except:
            print("Incomplete data:", cfg.save_path)

    return performance

def get_lr_curve(cfgs, idx, file_name='lr_info.pth'):

    num_seeds = len(cfgs)
    info = []

    for seed in range(num_seeds):
        cfg = cfgs[seed][idx]

        try:
            file_path = osp.join(cfg.save_path, file_name)
            mean, std = torch.load(file_path)
            info.append(np.array(mean))
        except:
            print("Incomplete data:", cfg.save_path)

    info = np.stack(info, axis=1)
    return info

def get_test_mean_ci(cfgs, idx, cfgs2=None, key='TestAcc'):
    performance = get_performance(
        cfgs, idx, 
        key1='TestLoss', key2=key, 
        file_name='test.txt', beg=0, 
        check_complete=(cfgs2 is not None), cfgs2=cfgs2
    )
    return np.mean(performance), plots.get_sem(performance) * 1.96

def cuereward_demo_analysis():
    
    cfgs = experiments.cuereward_demo()
    model_list = ['RNN', ]
    rule_list = ['gradient', 'hebbian', 'none', ]
    performance = []

    for i, model in enumerate(model_list):

        performance = []
        for k, rule in enumerate(rule_list):
            curve, x_axis = get_curve(cfgs, i * 3 + k, key='TestLoss', start=1)
            performance.append(curve)

        plots.error_range_plot(
            x_axis,
            performance,
            x_label='Training Step',
            y_label='Validation Loss',
            label_list=rule_list,
            fig_dir='cuereward',
            fig_name=f'curves_demo',
            fig_size=(5, 4)
        )

def seqreproduction_long_compare_delay_analysis():
    cfgs = experiments.seqreproduction_long_compare_delay()
    model_list = ['RNN', 'LSTM']
    rule_list = ['gradient', 'hebbian', 'none', ]

    for j, delay in enumerate([0, 20, 40]):

        for i, model in enumerate(model_list):

            performance = []
            for k, rule in enumerate(rule_list):

                curve, x_axis = get_curve(cfgs, (i + j * 2) * 3 + k, key='TestLoss', start=1)
                performance.append(curve)

            plots.error_range_plot(
                x_axis,
                performance,
                x_label='Training Step',
                y_label=f'{model} Validation Loss',
                label_list=rule_list,
                fig_dir='seq',
                fig_name=f'{model}_delay{delay}'
            )

def seqreproduction_lr_curves_analysis():
    cfgs = experiments.seqreproduction_long_compare_delay_test()

    model_list = ['LSTM', ]
    rule_list = ['gradient', 'hebbian', ]

    for i, model in enumerate(model_list):

        performance = []

        for k, rule in enumerate(rule_list):
            curve = get_lr_curve(cfgs, i * 2 + k)
            performance.append(curve)

        x_axis = np.arange(1, len(performance[0]) + 1)
        plots.error_range_plot(
            x_axis,
            performance,
            x_label='$t$',
            y_label=f'$\eta(t)$',
            label_list=rule_list,
            fig_dir='seq',
            fig_name=f'{model}_lr_curves',
            y_ticks=[0, 0.1, 0.2],
            fig_size=(5, 4)
        )

def seqreproduction_delay_analysis():

    cfgs = experiments.seqreproduction_delay()
    model_list = ['RNN', 'LSTM']
    rule_list = ['gradient', 'hebbian', 'none', ]
    x_axis = [0, 10, 20, 40, 60]

    for i, model in enumerate(model_list):

        performance = []

        for k, rule in enumerate(rule_list):

            performance.append([])
            for j, delay in enumerate(x_axis):

                res = get_performance(
                    cfgs, (i + j * 2) * 3 + k, 
                    key1='TestLoss',  
                    file_name='test.txt', beg=0,
                    check_complete=False
                )
                performance[-1].append(res)

        plots.errorbar_plot(
            x_axis,
            performance,
            x_label='Delay $m$',
            y_label=f'Test Loss',
            label_list=rule_list,
            fig_dir='seq',
            fig_name=f'{model}_compare_delay',
            add=0.0035,
            figsize=(5, 4)
        )

def seqreproduction_seqlength_analysis():

    cfgs = experiments.seqreproduction_seqlength()
    model_list = ['RNN', 'LSTM']
    rule_list = ['gradient', 'hebbian', 'none', ]
    x_axis = [5, 10, 20, 30, 40]

    for i, model in enumerate(model_list):

        performance = []

        for k, rule in enumerate(rule_list):

            performance.append([])
            for j, delay in enumerate(x_axis):

                res = get_performance(
                    cfgs, (i + j * 2) * 3 + k, 
                    key1='TestLoss',  
                    file_name='test.txt', beg=0,
                    check_complete=False
                )
                performance[-1].append(res)

        plots.errorbar_plot(
            x_axis,
            performance,
            x_label='Sequence Length $n$',
            y_label=f'Test Loss',
            label_list=rule_list,
            fig_dir='seq',
            fig_name=f'{model}_compare_seqlength',
            add=0.0035,
            figsize=(5, 4)
        )

def cuereward_lr_curves_analysis():

    cfgs = experiments.cuereward_large_test()

    model_list = ['RNN', 'LSTM', ]
    rule_list = ['gradient', 'hebbian', ]

    for i, model in enumerate(model_list):

        performance = []

        for k, rule in enumerate(rule_list):
            curve = get_lr_curve(cfgs, i * 2 + k)
            performance.append(curve)

        x_axis = np.arange(1, len(performance[0]) + 1)
        plots.error_range_plot(
            x_axis,
            performance,
            x_label='$t$',
            y_label=f'$\eta(t)$',
            label_list=rule_list,
            fig_dir='cuereward',
            fig_name=f'{model}_lr_curves',
            y_ticks=[0, 0.1, 0.2],
            fig_size=(5, 4)
        )

def regression_mlp_curves_analysis():

    cfgs = experiments.regression_mlp_test()

    model_list = ['RNN', 'LSTM', ]
    rule_list = ['gradient', 'hebbian', ]

    for i, model in enumerate(model_list):

        performance = []

        for k, rule in enumerate(rule_list):
            curve = get_lr_curve(cfgs, i * 2 + k)
            performance.append(curve)

        x_axis = np.arange(1, len(performance[0]) + 1)
        plots.error_range_plot(
            x_axis,
            performance,
            x_label='$t$',
            y_label=f'$\eta(t)$',
            label_list=rule_list,
            fig_dir='regression',
            fig_name=f'{model}_lr_curves_mlp',
            y_ticks=[0, 0.1, 0.2],
            fig_size=(5, 4)
        )

def cuereward_large_analysis():
    
    cfgs = experiments.cuereward_large()
    model_list = ['RNN', 'LSTM']
    rule_list = ['gradient', 'hebbian', 'none', ]
    performance = []

    for i, model in enumerate(model_list):

        performance = []
        for k, rule in enumerate(rule_list):

            curve, x_axis = get_curve(cfgs, i * 3 + k, key='TestLoss', start=1)
            performance.append(curve)

        plots.error_range_plot(
            x_axis,
            performance,
            x_label='Training Step',
            y_label='Validation Loss',
            label_list=rule_list,
            fig_dir='cuereward',
            fig_name=f'curves_{model}',
            fig_size=(5, 4)
        )

def cuereward_random_network_analysis():
    
    cfgs = experiments.cuereward_random_network()
    model_list = ['uniform', 'neg_uniform', 'random', ]

    for k, rule in enumerate(['gradient', 'hebbian']):
        for j, rnn in enumerate(['RNN', 'LSTM']):
            performance = []

            for i, model in enumerate(model_list):

                curve, x_axis = get_curve(cfgs, i + (j + k * 2) * 3, key='TestLoss', start=1)
                performance.append(curve)

            plots.error_range_plot(
                x_axis,
                performance,
                x_label='Training Step',
                y_label='Validation Loss',
                label_list=model_list,
                fig_dir='cuereward_rn',
                fig_name=f'inner_lr_{rule}_{rnn}'
            )


def cuereward_modulation_analysis():
    
    cfgs = experiments.cuereward_modulation()
    model_list = ['RNN-gradient', 'RNN-hebbian', 'LSTM-gradient', 'LSTM-hebbian']
    rule_list = ['modulated', 'non-modulated', ]
    performance = []

    for i, model in enumerate(model_list):

        performance = []
        for k, rule in enumerate(rule_list):
            curve, x_axis = get_curve(cfgs, i * 2 + k, key='TestLoss', start=1)
            performance.append(curve)

        plots.error_range_plot(
            x_axis,
            performance,
            x_label='Training Step',
            y_label='Validation Loss',
            label_list=rule_list,
            fig_dir='cuereward',
            fig_name=f'curves_modulation_{model}'
        )

def seqreproduction_modulation_analysis():
    
    cfgs = experiments.seqreproduction_modulation()
    model_list = ['RNN-gradient', 'RNN-hebbian', 'LSTM-gradient', 'LSTM-hebbian']
    rule_list = ['modulated', 'non-modulated', ]
    performance = []

    for i, model in enumerate(model_list):

        performance = []
        for k, rule in enumerate(rule_list):
            curve, x_axis = get_curve(cfgs, i * 2 + k, key='TestLoss', start=1)
            performance.append(curve)

        plots.error_range_plot(
            x_axis,
            performance,
            x_label='Training Step',
            y_label='Validation Loss',
            label_list=rule_list,
            fig_dir='seq',
            fig_name=f'curves_modulation_{model}'
        )

def regression_mlp_modulation_analysis():
    
    cfgs = experiments.regression_mlp_modulation()
    model_list = ['RNN-gradient', 'RNN-hebbian', 'LSTM-gradient', 'LSTM-hebbian']
    rule_list = ['modulated', 'non-modulated', ]
    performance = []

    for i, model in enumerate(model_list):

        performance = []
        for k, rule in enumerate(rule_list):
            curve, x_axis = get_curve(cfgs, i * 2 + k, key='TestLoss', start=1)
            performance.append(curve)

        plots.error_range_plot(
            x_axis,
            performance,
            x_label='Training Step',
            y_label='Validation Loss',
            label_list=rule_list,
            fig_dir='regression',
            fig_name=f'curves_modulation_mlp_{model}'
        )

def cuereward_inner_lr_mode_analysis():

    cfgs = experiments.cuereward_inner_lr_mode()
    model_list = ['none', 'uniform', 'neg_uniform', 'random']

    for k, rule in enumerate(['gradient', 'hebbian']):
        for j, rnn in enumerate(['RNN', 'LSTM']):
            performance = []

            for i, model in enumerate(model_list):

                curve, x_axis = get_curve(cfgs, i + (j + k * 2) * 4, key='TestLoss', start=1)
                performance.append(curve)

            plots.error_range_plot(
                x_axis,
                performance,
                x_label='Training Step',
                y_label='Validation Loss',
                label_list=model_list,
                fig_dir='cuereward',
                fig_name=f'inner_lr_{rule}_{rnn}'
            )

            if rule == 'gradient' and rnn == 'LSTM':
                plots.error_range_plot(
                    x_axis,
                    [performance[0], performance[-1], ],
                    x_label='Training Step',
                    y_label='Validation Loss',
                    label_list=['Without $\\alpha$', 'With $\\alpha$'],
                    fig_dir='cuereward',
                    fig_name=f'inner_lr_{rule}_{rnn}_extra'
                )

def cuereward_lr_analysis():
    cfgs = experiments.cuereward_lr()

    for idx, lr in enumerate([0.01, 0.0003]):
        model_list = ['RNN', 'LSTM']
        rule_list = ['gradient', 'hebbian', 'none', ]
        performance = []

        for i, model in enumerate(model_list):

            performance = []
            for k, rule in enumerate(rule_list):

                curve, x_axis = get_curve(cfgs, (idx * 2 + i) * 3 + k, key='TestLoss', start=1)
                performance.append(curve)

            plots.error_range_plot(
                x_axis,
                performance,
                x_label='Training Step',
                y_label='Validation Loss',
                label_list=rule_list,
                fig_dir='cuereward',
                fig_name=f'curves_{model}_lr{lr}',
                fig_size=(5, 4)
            )

def cuereward_extradim_analysis():
    cfgs = experiments.cuereward_extradim()

    models = ['RNN', 'LSTM', ]
    x_axis = [0, 4, 8, 16]

    for j, rule in enumerate(['gradient', 'hebbian']):

        performances = []
        for i, model in enumerate(models):

            performances.append([])
            for idx, d in enumerate(x_axis):
                performances[-1].append(
                    get_performance(
                        cfgs, (i + j * 2) * len(x_axis) + idx,
                        key1='TestLoss',  
                        file_name='test.txt', beg=0,
                        check_complete=False
                    )
                )

        plots.errorbar_plot(
            x_axis,
            performances,
            'dim',
            'Validation Error',
            models,
            fig_dir='cuereward',
            fig_name=f'extradim_{rule}'
        )

def cuereward_plr_analysis():
    cfgs = experiments.cuereward_plr()

    models = ['RNN', 'LSTM', ]
    x_axis = [0, 0.025, 0.05, 0.1, 0.2]
    x_axis = [x * 2 for x in x_axis]

    for j, rule in enumerate(['gradient', 'hebbian']):

        performances = []
        for i, model in enumerate(models):

            performances.append([])
            for idx, d in enumerate(x_axis):
                performances[-1].append(
                    get_performance(
                        cfgs, (i + j * 2) * len(x_axis) + idx, 
                        key1='TestLoss',  
                        file_name='test.txt', beg=0,
                        check_complete=False
                    )
                )

        plots.errorbar_plot(
            x_axis,
            performances,
            '$\eta_0$',
            'Validation Error',
            models,
            fig_dir='cuereward',
            fig_name=f'innerlr_{rule}'
        )

def regression_analysis():
    cfgs = experiments.regression()

    dataset_list = [10, 20]
    for i, dataset in enumerate(dataset_list):

        rule_list = ['gradient', 'hebbian', 'none', ]
        model_list = ['RNN', 'LSTM', ]
        performance = []

        print(dataset)

        for k, model in enumerate(model_list):
            for j, rule in enumerate(rule_list):
                performance.append(get_test_mean_ci(cfgs, (i * 2 + k) * 3 + j, key='TestError'))
                mean, ci = performance[-1]
                print(rule, model, ":", mean, ci)

def regression_mlp_analysis():
    cfgs = experiments.regression_mlp()

    dataset_list = [10, 20]
    for i, dataset in enumerate(dataset_list):

        rule_list = ['gradient', 'hebbian', 'none', ]
        model_list = ['RNN', 'LSTM', ]
        performance = []

        print(dataset)

        for k, model in enumerate(model_list):
            for j, rule in enumerate(rule_list):
                performance.append(get_test_mean_ci(cfgs, (i * 2 + k) * 3 + j, key='TestError'))
                mean, ci = performance[-1]
                print(rule, model, ":", mean, ci)

def fsc_analysis():
    cfgs = experiments.fsc()

    dataset_list = ['CIFAR_FS', 'miniImageNet']
    for i, dataset in enumerate(dataset_list):

        rule_list = ['gradient', 'hebbian', 'none', ]
        model_list = ['RNN', 'LSTM', ]
        performance = []

        print(dataset)

        for k, model in enumerate(model_list):
            for j, rule in enumerate(rule_list):
                performance.append(get_test_mean_ci(cfgs, (i * 3 + j) * 2 + k))
                mean, ci = performance[-1]
                print(rule, model, ":", mean, ci)


def fsc_resnet_analysis():
    cfgs = experiments.fsc_resnet()

    dataset_list = ['CIFAR_FS', 'miniImageNet']
    for i, dataset in enumerate(dataset_list):

        rule_list = ['gradient', 'hebbian', 'none', ]
        model_list = ['RNN', 'LSTM', ]
        performance = []

        print(dataset)

        for k, model in enumerate(model_list):
            for j, rule in enumerate(rule_list):
                performance.append(get_test_mean_ci(cfgs, (i * 3 + j) * 2 + k))
                mean, ci = performance[-1]
                print(rule, model, ":", mean, ci)

def fsc_curves_analysis():
    cfgs = experiments.fsc()

    dataset_list = ['CIFAR_FS', 'miniImageNet']
    for i, dataset in enumerate(dataset_list):

        rule_list = ['gradient', 'hebbian', 'none', ]
        model_list = ['RNN', 'LSTM', ]
        print(dataset)

        for k, model in enumerate(model_list):
            
            if model == 'LSTM':
                continue

            train_loss = []
            val_acc = []

            for j, rule in enumerate(rule_list):

                curve, x_axis = get_curve(cfgs, (i * 3 + j) * 2 + k, key='TrainLoss', start=1)
                train_loss.append(curve)

                curve, x_axis = get_curve(cfgs, (i * 3 + j) * 2 + k, key='TestAcc', start=1)
                val_acc.append(curve)

            plots.error_range_plot(
                x_axis,
                train_loss,
                x_label='Training Step',
                y_label='Training Loss',
                label_list=rule_list,
                fig_dir='fsc',
                fig_name=f'train_{model}_{dataset}_proto'
            )

            plots.error_range_plot(
                x_axis,
                val_acc,
                x_label='Training Step',
                y_label='Validation Accuracy',
                label_list=rule_list,
                fig_dir='fsc',
                fig_name=f'val_{model}_{dataset}_proto'
            )

def fsc_resnet_curves_analysis():
    cfgs = experiments.fsc_resnet()

    dataset_list = ['CIFAR_FS', 'miniImageNet']
    for i, dataset in enumerate(dataset_list):

        rule_list = ['gradient', 'hebbian', 'none', ]
        model_list = ['RNN', 'LSTM', ]
        print(dataset)

        for k, model in enumerate(model_list):
            
            if model == 'LSTM':
                continue

            train_loss = []
            val_acc = []

            for j, rule in enumerate(rule_list):

                curve, x_axis = get_curve(cfgs, (i * 3 + j) * 2 + k, key='TrainLoss', start=1)
                train_loss.append(curve)

                curve, x_axis = get_curve(cfgs, (i * 3 + j) * 2 + k, key='TestAcc', start=1)
                val_acc.append(curve)

            plots.error_range_plot(
                x_axis,
                train_loss,
                x_label='Training Step',
                y_label='Training Loss',
                label_list=rule_list,
                fig_dir='fsc',
                fig_name=f'train_{model}_{dataset}'
            )

            plots.error_range_plot(
                x_axis,
                val_acc,
                x_label='Training Step',
                y_label='Validation Accuracy',
                label_list=rule_list,
                fig_dir='fsc',
                fig_name=f'val_{model}_{dataset}'
            )
