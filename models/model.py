__all__ = ['SimpleRNN', 'CNNtoRNN', 'CNN', ]

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.configs import BaseConfig
from models.plastic.meta import PlasticParam

from utils.model_utils import get_cnn, get_rnn, get_linear
from configs.config_global import DEVICE
import models

class SimpleRNN(models.PlasticModule):

    def __init__(self, config: BaseConfig, custom_input=False):
        super().__init__()

        PlasticParam.set_elementwise_lr(config.inner_lr_mode)
        PlasticParam.set_param_grad(not config.random_network)

        assert len(config.input_shape) == 1 or custom_input, "input must be 1-dim for SimpleRNN"

        if not custom_input:
            self.in_fc = get_linear(
                config.plasticity_mode, 
                config.input_shape[0], 
                config.hidden_size, 
                activation='relu'
            )

        self.custom_input = custom_input

        self.rnn = get_rnn(config.rnn, config.plasticity_mode, config.hidden_size, config.hidden_size)
        self.full_outsize = config.model_outsize + config.extra_dim + config.modulation
        self.out_fc = get_linear(
            config.plasticity_mode, 
            config.hidden_size, 
            self.full_outsize
        )
        self.rnn_type = config.rnn

        self.plasticity_mode = config.plasticity_mode
        self.dim = self.get_floatparam_dim()
        self.hidden_size = config.hidden_size
        self.out_dim = config.model_outsize
        self.modulation = config.modulation
        self.out_weight = nn.Parameter(torch.ones((self.full_outsize, )))
               
        self.lr = config.p_lr
        self.wd = config.p_wd
        self.grad_clip = config.inner_grad_clip
        self.weight_clip = config.weight_clip

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

        loss = F.mse_loss(x * self.out_weight, torch.zeros_like(x))

        lr = torch.full_like(x[:, -1], self.lr)
        wd = torch.full_like(x[:, -1], self.wd)

        if self.modulation:
            lr = lr * torch.sigmoid(x[:, -1]) * 2
            wd = wd * torch.sigmoid(x[:, -1]) * 2

        if self.dim > 0:
            floatparam = self.update_floatparam(loss, lr, wd, self.grad_clip, mode=self.plasticity_mode)
            if self.weight_clip is not None:
                floatparam = torch.clip(floatparam, -self.weight_clip, self.weight_clip)
            h = torch.cat([floatparam, h], dim=1)

        info = dict(
            lr=lr.detach().cpu(),
            wd=wd.detach().cpu(),
            inner_loss=loss.detach().cpu()
        )

        return x[:, :self.out_dim], h, info

    def forward(self, input, hidden, detail=False):

        flag = torch.is_grad_enabled()
        with torch.enable_grad():
            x, h, info = self._forward(input, hidden)
            
        if not flag:
            x = x.detach()
            h = h.detach()

        if detail:
            return x, h, info
        else:
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
            get_linear("none", out_shape[1], config.hidden_size - config.extra_input_dim, fan_in=True),
            nn.ReLU()
        )

    def forward(self, input, hidden, **kwargs):

        img_input, extra_input = input
        embedding = self.proj(self.cnn(img_input))
        embedding = torch.cat((embedding, extra_input), dim=1)
        return super().forward(embedding, hidden, **kwargs)

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