__all__ = ['SimpleRNN', 'CNNtoRNN', 'CNN', 'RecurrentPolicy']

from copy import deepcopy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.configs import BaseConfig
from models.plastic.meta import PlasticParam

from utils.model_utils import get_cnn, get_rnn, get_linear
import models
import torch_ac as ac

class SimpleRNN(models.PlasticModule):

    def __init__(self, config: BaseConfig, custom_input=False):
        super().__init__()

        PlasticParam.set_elementwise_lr(config.inner_lr_mode)

        assert len(config.input_shape) == 1 or custom_input, "input must be 1-dim for SimpleRNN"

        if not custom_input:
            self.in_fc = nn.Sequential(
                get_linear(config.plasticity_mode, config.input_shape[0], config.hidden_size),
                nn.ReLU(),
            )
        self.custom_input = custom_input

        self.rnn = get_rnn(config.rnn, config.plasticity_mode, config.hidden_size, config.hidden_size)
        self.out_fc = get_linear(config.plasticity_mode, config.hidden_size, config.model_outsize + config.extra_dim)
        self.rnn_type = config.rnn

        self.dim = self.get_floatparam_dim()
        self.hidden_size = config.hidden_size
        self.out_dim = config.model_outsize
        
        self.lr = config.p_lr
        self.wd = config.p_wd
        self.grad_clip = config.inner_grad_clip

        self.use_layernorm = config.layernorm
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm((self.hidden_size, ))

    def _forward(self, input, hidden):

        if self.dim > 0:
            self.set_floatparam(hidden[:, :self.dim])

        if not self.custom_input:
            x = self.in_fc(input)
        else:
            x = input

        if self.rnn_type == 'LSTM':
            x, h = self.rnn(x, (hidden[:, self.dim: -self.hidden_size], hidden[:, -self.hidden_size: ]))
            if self.use_layernorm:
                x = self.layernorm(x)
            h = torch.cat((x, h), dim=1)
            x = self.out_fc(x)
        else:
            h = self.rnn(x, hidden[:, self.dim: ])
            if self.use_layernorm:
                h = self.layernorm(h)
            x = self.out_fc(h)

        loss = F.mse_loss(x, torch.zeros_like(x))

        if self.dim > 0:
            floatparam = self.update_floatparam(loss, self.lr, self.wd, self.grad_clip)
            h = torch.cat([floatparam, h], dim=1)

        return x[:, :self.out_dim], h

    def forward(self, input, hidden):

        flag = torch.is_grad_enabled()
        with torch.enable_grad():
            x, h = self._forward(input, hidden)
            
        if not flag:
            x = x.detach()
            h = h.detach()

        return x, h

    @property
    def memory_size(self):
        return self.dim + self.hidden_size * (1 + (self.rnn_type == 'LSTM'))

class CNNtoRNN(SimpleRNN):

    def __init__(self, config: BaseConfig):
        assert len(config.input_shape) == 3, "input must be 3-dim for CNNtoRNN"

        super().__init__(config, custom_input=True)

        self.cnn = get_cnn(config.cnn, config)
        out_shape = self.cnn(torch.rand(config.input_shape).unsqueeze(0)).shape

        self.proj = nn.Sequential(
            get_linear("none", out_shape[1], config.hidden_size - config.extra_input_dim),
            nn.ReLU()
        )

    def forward(self, input, hidden):

        img_input, extra_input = input
        embedding = self.proj(self.cnn(img_input))
        embedding = torch.cat((embedding, extra_input), dim=1)
        return super().forward(embedding, hidden)

class RecurrentPolicy(SimpleRNN):

    recurrent = True

    def __init__(self, config: BaseConfig):

        super().__init__(config, custom_input=True)

        if len(config.input_shape) == 3:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, (7, 7), 4),
                nn.ReLU(),
                nn.Conv2d(16, 32, (3, 3), 2),
                nn.ReLU(),
                nn.Conv2d(32, 64, (3, 3), 2),
                nn.ReLU(),
                nn.Flatten()
            )
            input_size = self.encoder(torch.rand(config.input_shape).unsqueeze(0)).shape[1]
        elif len(config.input_shape) == 1:
            self.encoder = nn.Identity()
            input_size = config.input_shape[0]
        else:
            raise ValueError(config.input_shape)

        self.proj = nn.Sequential(
            get_linear("none", input_size, config.hidden_size),
            nn.ReLU()
        )

    def forward(self, input, hidden):
        embedding = self.proj(self.encoder(input))
        out, hidden = super().forward(embedding, hidden)
        value = out[:, 0]
        dist = torch.distributions.Categorical(logits=out[:, 1: ])
        return dist, value, hidden

class CNN(nn.Module):

    def __init__(self, config: BaseConfig):
        assert len(config.input_shape) == 3, "input must be 3-dim for CNNtoRNN"

        super().__init__()
        self.cnn = get_cnn(config.cnn, config)
        out_shape = self.cnn(torch.rand(config.input_shape).unsqueeze(0)).shape

        if hasattr(config, "use_mlp_proj") and config.use_mlp_proj:
            self.proj = nn.Sequential(
                nn.Linear(out_shape[1], config.model_outsize),
                nn.ReLU(),
                nn.Linear(config.model_outsize, config.model_outsize)
            )
        else:
            self.proj = nn.Linear(out_shape[1], config.model_outsize)

    def forward(self, input):
        embedding = self.proj(self.cnn(input))
        return embedding