
__all__ = ['HebbLinear', 'PlasticLinear', 'Linear']

from numpy import kaiser
from ..plastic import *
from .activation import *

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class PlasticLinear(nn.Module):

    def __init__(self, ind, outd, scale=1, fan_in=False):
        super().__init__()
        self.w = PlasticParam(torch.empty(ind, outd))
        self.b = PlasticParam(torch.empty(outd))
        if fan_in:
            self.InitWeight(scale / math.sqrt(ind))
        else:
            self.InitWeight(scale / math.sqrt(outd))

    def InitWeight(self, k):
        init.uniform_(self.w.param.data, -k, k)
        init.uniform_(self.b.param.data, -k, k)
        self.init_k = k
       
    def forward(self, x):
        w = self.w() # bsz * in * out
        b = self.b().unsqueeze(1) # bsz * out
        x = x.unsqueeze(1) # bsz * 1 * in
        # print(w.shape, b.shape, x.shape)
        return torch.baddbmm(b, x, w).squeeze(1)

class HebbLinear(PlasticLinear):

    def __init__(self, ind, outd, scale=1, fan_in=False, activation='none'):
        super().__init__(ind, outd, scale, fan_in)
        self.non_linearity = Activation(activation)

    def forward(self, x):
        self.w.pre = x
        out = super().forward(x)
        out = self.non_linearity(out)
        self.w.post = out
        return out

class Linear(nn.Linear):

    def __init__(self, ind, outd, scale=1, fan_in=False):
        super().__init__(ind, outd)
        if fan_in:
            self.InitWeight(scale / math.sqrt(ind))
        else:
            self.InitWeight(scale / math.sqrt(outd))
        
    def InitWeight(self, k):
        init.uniform_(self.weight.data, -k, k)
        init.uniform_(self.bias.data, -k, k)
        self.init_k = k