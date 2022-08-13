
__all__ = ['Activation']

import torch
import torch.nn as nn

class Activation(nn.Module):

    def __init__(self, type, **kwargs):
        super().__init__()

        if type == 'relu':
            self.act = nn.ReLU(**kwargs)
        elif type == 'tanh':
            self.act = nn.Tanh(**kwargs)
        elif type == 'sigmoid':
            self.act = nn.Sigmoid(**kwargs)
        elif type == 'none':
            self.act = nn.Identity()
        else:
            raise(ValueError(type))

    def forward(self, x):
        return self.act(x)