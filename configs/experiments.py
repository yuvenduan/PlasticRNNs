"""Experiments and corresponding analysis.
format adapted from https://github.com/gyyang/olfaction_evolution

Each experiment is described by a function that returns a list of configurations
function name is the experiment name
"""

from collections import OrderedDict
from configs.configs import BaseConfig, ClassificationPretrainConfig, CueRewardConfig, FSCConfig, RegressionConfig, SeqReproductionConfig, ContrastivePretrainConfig
from utils.config_utils import vary_config
from copy import deepcopy

import os.path as osp

def scale_modelsize(configs, factor=1.5):
    for key, cfgs in configs.items():
        for cfg in cfgs:
            if cfg.plasticity_mode == 'none' or cfg.inner_lr_mode == 'none':
                cfg.hidden_size = int(cfg.hidden_size * factor)

def scale_rnn_modelsize(configs, factor=2):
    for key, cfgs in configs.items():
        for cfg in cfgs:
            if cfg.rnn == 'RNN':
                cfg.hidden_size = int(cfg.hidden_size * factor)

def set_equal_lr_wd(configs):
    for key, cfgs in configs.items():
        for cfg in cfgs:
            cfg.p_wd = cfg.p_lr

def cuereward_demo():
    config = CueRewardConfig()

    config.experiment_name = 'cuereward_demo'
    config.max_batch = 5000

    config_ranges = OrderedDict()
    config_ranges['rnn'] = ['RNN', ]
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', 'none',  ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=1)
    scale_modelsize(configs, 1.5)

    return configs

def cuereward_lr():
    config = CueRewardConfig()

    config.experiment_name = 'cuereward_lr'
    
    config_ranges = OrderedDict()
    config_ranges['lr'] = [0.01, 0.0003]
    config_ranges['rnn'] = ['RNN', 'LSTM', ]
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', 'none',  ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs, 1.5)
    scale_rnn_modelsize(configs)
    return configs

def cuereward_plr():
    config = CueRewardConfig()

    config.experiment_name = 'cuereward_plr'

    config_ranges = OrderedDict()
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian']
    config_ranges['rnn'] = ['RNN', 'LSTM', ]
    config_ranges['p_lr'] = [0, 0.025, 0.05, 0.1, 0.2, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    set_equal_lr_wd(configs)
    scale_rnn_modelsize(configs)
    return configs

def cuereward_extradim():
    config = CueRewardConfig()
    config.experiment_name = 'cuereward_extradim'

    config_ranges = OrderedDict()
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian']
    config_ranges['rnn'] = ['RNN', 'LSTM', ]
    config_ranges['extra_dim'] = [0, 4, 8, 16]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    scale_modelsize(configs, 1.5)
    scale_rnn_modelsize(configs)

    return configs

def cuereward_large():
    config = CueRewardConfig()

    config.experiment_name = 'cuereward_large'

    config_ranges = OrderedDict()
    config_ranges['rnn'] = ['RNN', 'LSTM', ]
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', 'none',  ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    scale_modelsize(configs, 1.5)
    scale_rnn_modelsize(configs)

    return configs

def cuereward_random_network():
    config = CueRewardConfig()

    config.random_network = True
    config.experiment_name = 'cuereward_random_network'

    config_ranges = OrderedDict()
    config_ranges['rnn'] = ['RNN', 'LSTM', ]
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', ]
    config_ranges['inner_lr_mode'] = ['uniform', 'neg_uniform', 'random']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    scale_modelsize(configs, 1.5)
    scale_rnn_modelsize(configs)

    return configs

def cuereward_noisy():
    config = CueRewardConfig()
    config.input_noise = 0.5
    config.experiment_name = 'cuereward_noisy'

    config_ranges = OrderedDict()
    config_ranges['rnn'] = ['RNN', 'LSTM', ]
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', 'none',  ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    scale_modelsize(configs, 1.5)
    scale_rnn_modelsize(configs)
    return configs

def cuereward_modulation():
    config = CueRewardConfig()
    config.experiment_name = 'cuereward_modulation'

    config_ranges = OrderedDict()
    config_ranges['rnn'] = ['RNN', 'LSTM', ]
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', ]
    config_ranges['modulation'] = [True, False]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    scale_modelsize(configs, 1.5)
    scale_rnn_modelsize(configs)

    return configs

def cuereward_inner_lr_mode():
    config = CueRewardConfig()
    config.experiment_name = 'cuereward_inner_lr_mode'

    config_ranges = OrderedDict()
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian']
    config_ranges['rnn'] = ['RNN', 'LSTM', ]
    config_ranges['inner_lr_mode'] = ['none', 'uniform', 'neg_uniform', 'random']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    return configs

def seqreproduction_long_compare_delay():
    config = SeqReproductionConfig()
    config.experiment_name = 'seqreproduction_long_compare_delay'
    config.gpu_constraint = '18GB'

    config_ranges = OrderedDict()
    config_ranges['delay'] = [0, 20, 40]
    config_ranges['rnn'] = ['RNN', 'LSTM', ]
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', 'none', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs)
    scale_rnn_modelsize(configs)
    return configs

def seqreproduction_modulation():
    config = SeqReproductionConfig()
    config.experiment_name = 'seqreproduction_modulation'
    config.gpu_constraint = '18GB'

    config_ranges = OrderedDict()
    config_ranges['delay'] = [40, ]
    config_ranges['rnn'] = ['RNN', 'LSTM', ]
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', ]
    config_ranges['modulation'] = [True, False]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs)
    scale_rnn_modelsize(configs)
    return configs

def seqreproduction_delay():
    config = SeqReproductionConfig()
    config.experiment_name = 'seqreproduction_delay'

    config_ranges = OrderedDict()
    config_ranges['delay'] = [0, 10, 20, 40, 60]
    config_ranges['seq_length'] = [5, ]
    config_ranges['rnn'] = ['RNN', 'LSTM', ]
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', 'none', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs)
    scale_rnn_modelsize(configs)

    for key, cfgs in configs.items():
        for cfg in cfgs:
            if cfg.delay >= 40 and cfg.plasticity_mode != 'none':
                cfg.gpu_constraint = '18GB'

    return configs

def seqreproduction_seqlength():
    config = SeqReproductionConfig()
    config.experiment_name = 'seqreproduction_seqlength'
    config.gpu_constraint = '18GB'

    config_ranges = OrderedDict()
    config_ranges['delay'] = [0, ]
    config_ranges['seq_length'] = [5, 10, 20, 30, 40]
    config_ranges['rnn'] = ['RNN', 'LSTM', ]
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', 'none', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs)
    scale_rnn_modelsize(configs)

    for key, cfgs in configs.items():
        for cfg in cfgs:
            if cfg.seq_length >= 30 and cfg.plasticity_mode != 'none':
                cfg.gpu_constraint = '18GB'

    return configs

def seqreproduction_long_compare_delay_test():
    configs = seqreproduction_long_compare_delay()
    new_configs = {}

    for key, cfgs in configs.items():
        new_configs[key] = []

        for config in cfgs:
            for delay in [40, ]:
                if config.delay != 40 or config.rnn != 'LSTM' or config.plasticity_mode == 'none':
                    continue

                cfg: SeqReproductionConfig = deepcopy(config)
                cfg.delay = delay
                cfg.load_path = osp.join(config.save_path, 'net_best.pth')
                cfg.experiment_name = config.experiment_name + '_test'
                cfg.save_path = config.save_path.replace(config.experiment_name, cfg.experiment_name) + f'_delay{delay}'
                cfg.max_batch = 0
                cfg.do_analysis = True
                new_configs[key].append(cfg)

    return new_configs

def seqreproduction_norm():
    config = SeqReproductionConfig()
    config.experiment_name = 'seqreproduction_norm'
    config.delay = 40
    config.seq_length = 5
    
    config_ranges = OrderedDict()
    config_ranges['rnn'] = ['RNN', ]
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', ]
    config_ranges['inner_grad_clip'] = [1, 100, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs)
    scale_rnn_modelsize(configs)

    for key, cfgs in configs.items():
        for cfg in cfgs:
            if cfg.delay >= 40 and cfg.plasticity_mode != 'none':
                cfg.gpu_constraint = '18GB'

    return configs

def seqreproduction_weight_clip():
    config = SeqReproductionConfig()
    config.experiment_name = 'seqreproduction_weight_clip'
    config.delay = 40
    config.seq_length = 5
    
    config_ranges = OrderedDict()
    config_ranges['rnn'] = ['RNN', ]
    config_ranges['plasticity_mode'] = ['hebbian', ]
    config_ranges['weight_clip'] = [0.01, 0.1, 1, ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs)
    scale_rnn_modelsize(configs)

    for key, cfgs in configs.items():
        for cfg in cfgs:
            if cfg.delay >= 40 and cfg.plasticity_mode != 'none':
                cfg.gpu_constraint = '18GB'

    return configs

def cuereward_large_test():
    configs = cuereward_large()
    new_configs = {}

    for key, cfgs in configs.items():
        new_configs[key] = []

        for config in cfgs:
            if config.plasticity_mode == 'none':
                continue
            cfg = deepcopy(config)
            cfg.load_path = osp.join(config.save_path, 'net_best.pth')
            cfg.experiment_name = config.experiment_name + '_test'
            cfg.save_path = config.save_path.replace(config.experiment_name, cfg.experiment_name)
            cfg.max_batch = 0
            cfg.do_analysis = True
            new_configs[key].append(cfg)

    return new_configs

def regression_mlp_test():
    configs = regression_mlp()
    new_configs = {}

    for key, cfgs in configs.items():
        new_configs[key] = []

        for config in cfgs:
            if config.plasticity_mode == 'none' or config.train_length != 20:
                continue

            cfg = deepcopy(config)
            cfg.load_path = osp.join(config.save_path, 'net_best.pth')
            cfg.experiment_name = config.experiment_name + '_test'
            cfg.save_path = config.save_path.replace(config.experiment_name, cfg.experiment_name)
            cfg.max_batch = 0
            cfg.do_analysis = True
            new_configs[key].append(cfg)

    return new_configs

def regression():
    config = RegressionConfig()
    config.experiment_name = 'regression'
    config.input_shape = (14, )

    config_ranges = OrderedDict()
    config_ranges['train_length'] = [10, 20]
    config_ranges['rnn'] = ['RNN', 'LSTM', ]
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', 'none', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs)
    scale_rnn_modelsize(configs)
    return configs

def regression_mlp():
    config = RegressionConfig()
    config.experiment_name = 'regression_mlp'
    
    config.input_shape = (8, )
    config.task_mode = 'mlp'

    config_ranges = OrderedDict()
    config_ranges['train_length'] = [10, 20]
    config_ranges['rnn'] = ['RNN', 'LSTM', ]
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', 'none', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs)
    scale_rnn_modelsize(configs)
    return configs

def regression_mlp_modulation():
    config = RegressionConfig()
    config.experiment_name = 'regression_mlp_modulation'
    
    config.input_shape = (8, )
    config.task_mode = 'mlp'

    config_ranges = OrderedDict()
    config_ranges['train_length'] = [20, ]
    config_ranges['rnn'] = ['RNN', 'LSTM', ]
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', ]
    config_ranges['modulation'] = [True, False]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs)
    scale_rnn_modelsize(configs)
    return configs

def fsc():
    config = FSCConfig()
    config.experiment_name = 'fsc'

    config_ranges = OrderedDict()
    config_ranges['image_dataset'] = ['CIFAR_FS', 'miniImageNet', ]
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', 'none', ]
    config_ranges['rnn'] = ['RNN', 'LSTM', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs)
    scale_rnn_modelsize(configs)
    configure_image_dataset(configs)

    return configs

def fsc_resnet():
    config = FSCConfig()
    config.experiment_name = 'fsc_resnet'
    config.cnn = 'ResNet'
    config.gpu_constraint = '18GB'
    config.hours = 48

    config_ranges = OrderedDict()
    config_ranges['image_dataset'] = ['CIFAR_FS', 'miniImageNet']
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', 'none', ]
    config_ranges['rnn'] = ['RNN', 'LSTM']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs)
    scale_rnn_modelsize(configs)
    configure_image_dataset(configs)

    for key, cfgs in configs.items():
        for cfg in cfgs:
            if cfg.image_dataset == 'miniImageNet':
                cfg.gpu_constraint = '80GB'

    return configs

def fsc_simclr_pretrain():
    config = FSCConfig()
    config.experiment_name = 'fsc_simclr_pretrain'
    config.cnn_pretrain = 'contrastive'
    config.pretrain_step = 100000
    config.freeze_pretrained_cnn = False
    config.gpu_constraint = '18GB'

    config_ranges = OrderedDict()
    config_ranges['image_dataset'] = ['miniImageNet', 'CIFAR_FS', ]
    config_ranges['cnn'] = ['ResNet', ]
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', 'none', ]
    config_ranges['rnn'] = ['LSTM', 'RNN']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs)
    scale_rnn_modelsize(configs)
    configure_image_dataset(configs)

    for key, cfgs in configs.items():
        for cfg in cfgs:
            if cfg.image_dataset == 'miniImageNet':
                cfg.gpu_constraint = '80GB'

    return configs

def configure_image_dataset(configs, set_outsize=False):
    for seed, cfgs in configs.items():
        for config in cfgs:
            config: ClassificationPretrainConfig

            if config.image_dataset == 'miniImageNet':
                if set_outsize:
                    config.model_outsize = 64
                config.input_shape = (3, 84, 84)
            elif config.image_dataset == 'CIFAR_FS':
                if set_outsize:
                    config.model_outsize = 100
                config.input_shape = (3, 32, 32)
            else:
                raise NotImplementedError("Image Dataset Not Implemented")

def classification_pretrain():
    config = ClassificationPretrainConfig()
    config.experiment_name = 'classification_pretrain'

    config.gpu_constraint = '11GB'
    config.save_every = 1000
    config.max_batch = 10000

    config_ranges = OrderedDict()
    config_ranges['image_dataset'] = ['miniImageNet', 'CIFAR_FS', ]
    config_ranges['cnn'] = ['ResNet', 'ProtoNet']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    configure_image_dataset(configs, True)

    return configs

def contrastive_pretrain():
    config = ContrastivePretrainConfig()
    config.experiment_name = 'contrastive_pretrain'
    config.gpu_constraint = '11GB'

    config_ranges = OrderedDict()
    config_ranges['image_dataset'] = ['miniImageNet', 'CIFAR_FS', ]
    config_ranges['cnn'] = ['ResNet', 'ProtoNet']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    configure_image_dataset(configs)

    for key, cfgs in configs.items():
        for cfg in cfgs:
            if cfg.image_dataset == 'miniImageNet' and cfg.cnn == 'ResNet':
                cfg.gpu_constraint = '18GB'

    return configs