"""Experiments and corresponding analysis.
format adapted from https://github.com/gyyang/olfaction_evolution

Each experiment is described by a function that returns a list of configurations
function name is the experiment name

combinatorial mode:
    config_ranges should not have repetitive values
sequential mode:
    config_ranges values should have equal length,
    otherwise this will only loop through the shortest one
control mode:
    base_config must contain keys in config_ranges
"""

from collections import OrderedDict
from turtle import delay
from configs.configs import BaseConfig, ClassificationPretrainConfig, CueRewardConfig, FSCConfig, RLBaseConfig, SeqReproductionConfig, ContrastivePretrainConfig
from utils.config_utils import vary_config

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

def cuereward():
    config = CueRewardConfig()

    config.experiment_name = 'cuereward'
    config.n_class = 2
    config.trial_length = 10
    config.max_batch = 10000

    config_ranges = OrderedDict()
    config_ranges['rnn'] = ['RNN', 'LSTM', ]
    config_ranges['plasticity_mode'] = ['hebbian', 'gradient', 'none', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs)
    scale_rnn_modelsize(configs)
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

def seqreproduction_short_compare_delay():
    config = SeqReproductionConfig()
    config.experiment_name = 'seqreproduction_short_compare_delay'

    config.seq_length = 10
    config_ranges = OrderedDict()
    config_ranges['delay'] = [0, 10, 20, 30,]
    config_ranges['rnn'] = ['RNN', 'LSTM', ]
    config_ranges['plasticity_mode'] = ['gradient', 'none',  ]

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

def fsc_large():
    config = FSCConfig()
    config.experiment_name = 'fsc_large'
    config.hidden_size = 512
    config.gpu_constraint = '18GB'

    config_ranges = OrderedDict()
    config_ranges['image_dataset'] = ['CIFAR_FS', 'miniImageNet', ]
    config_ranges['plasticity_mode'] = ['gradient', 'none', ]
    config_ranges['rnn'] = ['LSTM', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    scale_modelsize(configs)
    configure_image_dataset(configs)

    return configs

def fsc_resnet():
    config = FSCConfig()
    config.experiment_name = 'fsc_resnet'
    config.cnn = 'ResNet'
    config.gpu_constraint = '18GB'

    config_ranges = OrderedDict()
    config_ranges['image_dataset'] = ['CIFAR_FS', 'miniImageNet']
    config_ranges['plasticity_mode'] = ['gradient', 'none', ]
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

def fsc_lr():
    config = FSCConfig()
    config.experiment_name = 'fsc_lr'

    config_ranges = OrderedDict()
    config_ranges['image_dataset'] = ['miniImageNet', ]
    config_ranges['lr'] = [0.01, 0.003, 0.0003]
    config_ranges['plasticity_mode'] = ['gradient', 'none', ]
    config_ranges['rnn'] = ['LSTM', 'RNN']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs)
    scale_rnn_modelsize(configs)
    configure_image_dataset(configs)

    return configs

def fsc_wd():
    config = FSCConfig()
    config.experiment_name = 'fsc_wd'

    config_ranges = OrderedDict()
    config_ranges['image_dataset'] = ['miniImageNet', ]
    config_ranges['wdecay'] = [0.01, 0.002, 0.0001]
    config_ranges['plasticity_mode'] = ['gradient', 'none', ]
    config_ranges['rnn'] = ['LSTM', 'RNN']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs)
    scale_rnn_modelsize(configs)
    configure_image_dataset(configs)

    return configs

def fsc_plr():
    config = FSCConfig()
    config.experiment_name = 'fsc_plr'

    config_ranges = OrderedDict()
    config_ranges['image_dataset'] = ['miniImageNet', ]
    config_ranges['p_lr'] = [0, 0.03, 0.1, 0.2]
    config_ranges['plasticity_mode'] = ['gradient', ]
    config_ranges['rnn'] = ['LSTM', 'RNN']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_rnn_modelsize(configs)
    configure_image_dataset(configs)
    set_equal_lr_wd(configs)
    return configs

def fsc_extradim():
    config = FSCConfig()
    config.experiment_name = 'fsc_extradim'

    config_ranges = OrderedDict()
    config_ranges['image_dataset'] = ['miniImageNet', ]
    config_ranges['extra_dim'] = [0, 4, 16, 64]
    config_ranges['plasticity_mode'] = ['gradient', ]
    config_ranges['rnn'] = ['LSTM', 'RNN']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_rnn_modelsize(configs)
    configure_image_dataset(configs)
    return configs

def fsc_inner_lr_mode():
    config = FSCConfig()
    config.experiment_name = 'fsc_inner_lr_mode'

    config_ranges = OrderedDict()
    config_ranges['image_dataset'] = ['CIFAR_FS', 'miniImageNet', ]
    config_ranges['plasticity_mode'] = ['gradient', 'hebbian', ]
    config_ranges['rnn'] = ['RNN', ]
    config_ranges['inner_lr_mode'] = ['none', 'uniform', 'random']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    scale_rnn_modelsize(configs)
    configure_image_dataset(configs)
    return configs

def fsc_pretrain():
    config = FSCConfig()
    config.experiment_name = 'fsc_pretrain'
    config.cnn_pretrain = 'classification'
    config.pretrain_step = 5000

    config_ranges = OrderedDict()
    config_ranges['image_dataset'] = ['CIFAR_FS', 'miniImageNet', ]
    config_ranges['plasticity_mode'] = ['gradient', 'none', ]
    config_ranges['rnn'] = ['LSTM', 'RNN']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs)
    scale_rnn_modelsize(configs)
    configure_image_dataset(configs)

    return configs

def fsc_simclr_pretrain():
    config = FSCConfig()
    config.max_batch = 30000
    config.experiment_name = 'fsc_simclr_pretrain'
    config.cnn_pretrain = 'contrastive'
    config.pretrain_step = 100000

    config_ranges = OrderedDict()
    config_ranges['image_dataset'] = ['miniImageNet', 'CIFAR_FS', ]
    config_ranges['cnn'] = ['ResNet', 'ProtoNet']
    config_ranges['plasticity_mode'] = ['gradient', 'none', ]
    config_ranges['rnn'] = ['LSTM', 'RNN']

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)

    scale_modelsize(configs)
    scale_rnn_modelsize(configs)
    configure_image_dataset(configs)

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

def atari():
    config = RLBaseConfig()

    config.experiment_name = 'atari'
    config.model_outsize = 19

    config.env_kwargs = dict(full_action_space=True, frameskip=1)
    config.input_shape = (1, 84, 84)
    
    config.hidden_size = 128
    config.recurrence = 100
    config.horizon = 100
    config.log_every = 50

    config.algo = 'a2c'
    config.max_batch = 6250
    config.inner_grad_clip = 1
    config.extra_dim = 16

    config_ranges = OrderedDict()
    config_ranges['env'] = ["ALE/Alien-v5", "ALE/Pong-v5", "ALE/Amidar-v5"]
    config_ranges['plasticity_mode'] = ['gradient', 'none', ]
    config_ranges['rnn'] = ['RNN', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    scale_modelsize(configs)
    
    return configs

def atari_ram():
    config = RLBaseConfig()

    config.experiment_name = 'atari_ram'
    config.model_outsize = 19

    config.env_kwargs = dict(full_action_space=True, obs_type='ram')
    config.input_shape = (128, )
    
    config.hidden_size = 128
    config.recurrence = 100
    config.horizon = 100
    config.log_every = 50

    config.algo = 'a2c'
    config.max_batch = 10000
    config.layernorm = True
    config.extra_dim = 16
    config.clip_reward = True

    config_ranges = OrderedDict()
    config_ranges['env'] = ["ALE/Alien-v5", "ALE/Pong-v5", "ALE/Amidar-v5"]
    config_ranges['plasticity_mode'] = ['gradient', 'none', ]
    config_ranges['rnn'] = ['RNN', ]

    configs = vary_config(config, config_ranges, mode='combinatorial', num_seed=4)
    scale_modelsize(configs)
    
    return configs

def ant():
    config = RLBaseConfig()

    config.experiment_name = 'ant'
    config.env = "Ant-v4"

    config.model_outsize = 9
    config.input_shape = (27, )