
__all__ = ['InnerLayer', 'HebbLinear', 'PlasticLinear', 'Linear']

from numpy import kaiser
from ..plastic import *
from .activation import *

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class InnerAddition(nn.Module):

    def __init__(self, shape, scale=1):
        super().__init__()
        self.innerBias = PlasticParam(torch.zeros( shape))
        self.scale = scale

    def forward(self, x):
        # batchSize = len(x)
        # print(x.shape, batchSize, self.innerBias().shape)
        return (x + self.innerBias()) * self.scale

class InnerAddmul(nn.Module):

    def __init__(self, shape, scale=1):
        super().__init__()
        self.innerWeight = PlasticParam(torch.ones(shape))
        self.innerBias = PlasticParam(torch.zeros(shape))
        self.scale = scale

    def forward(self, x):
        # batchSize = len(x)
        return (x * self.innerWeight() + self.innerBias()) * self.scale

class InnerResmul(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.innerWeight = PlasticParam(torch.zeros(shape))

    def forward(self, x):
        # batchSize = len(x)
        return x * self.innerWeight() + x

class InnerAddresmul(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.innerWeight = PlasticParam(torch.zeros(shape))
        self.innerBias = PlasticParam(torch.zeros(shape))

    def forward(self, x):
        # batchSize = len(x)
        return x * self.innerWeight() + x + self.innerBias()

class InnerAddzeromul(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.innerWeight = PlasticParam(torch.zeros(shape))
        self.innerBias = PlasticParam(torch.zeros(shape))

    def forward(self, x):
        # batchSize = len(x)
        return x * self.innerWeight() + self.innerBias()

INNER_MODULE = {'none': nn.Identity, 'add': InnerAddition, 'addm': InnerAddmul, 'resm': InnerResmul, 'addresm': InnerAddresmul, 'addzerom': InnerAddzeromul}

class InnerLayer( nn.Module):

    def __init__(self, shape, type, **kwargs):
        super().__init__()
        self.module = INNER_MODULE[type](shape, **kwargs)

    def forward(self, x):
        return self.module(x.transpose(0, 1)).transpose(0, 1)

class HebbLinear( nn.Module):

    def __init__(self, ind, outd, scale=1):
        super().__init__()
        self.w = nn.Parameter(torch.empty(ind, outd))
        self.b = nn.Parameter(torch.empty(outd))
        self.alpha = nn.Parameter(torch.zeros(ind))
        self.hebb = None
        self.delta = None
        self.InitWeight(scale / math.sqrt(outd))

    def InitWeight(self, k):
        init.normal_(self.w.data, 0, k)
        init.uniform_(self.b.data, -k, k)
        # init.normal_(self.alpha.data, 0, k)

    def clearHebb(self):
        self.hebb = None
        self.delta = None

    @classmethod
    def clearAllHebb(cls, modules):
        for m in modules:
            if m.__class__ == cls:
                m.clearHebb()

    def step(self, eta):
        self.hebb = torch.clamp(self.hebb + self.delta * eta.unsqueeze(-1).unsqueeze(-1), -1, 1)

    @classmethod
    def Hebbstep(cls, modules, eta):
        for m in modules:
            if m.__class__ == cls:
                m.step(eta)

    def forward(self, x):
        if self.hebb is None:
            self.hebb = torch.zeros((x.shape[1], *self.w.shape), device=x.device)
        
        w = self.w + self.hebb * self.alpha.unsqueeze(-1)
        blk, bsz, ind = x.shape
        x = x.transpose(0, 1).reshape(-1, ind).unsqueeze(1) # bsz_blk * 1 * ind

        w = w.reshape(bsz * blk, ind, -1) # bsz_blk * ind * outd
        b = self.b.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * blk, 1, -1)
        # print(b.shape, w.shape, x.shape)

        y = torch.baddbmm(b, x, w) # bsz_blk * 1 * outd
        self.delta = torch.bmm(x.view(bsz * blk, ind, 1), y).reshape(bsz, blk, ind, -1) # bsz * blk * ind * outd
        # print(self.delta.mean(), self.delta.max())
        # exit(0)
        
        return y.reshape(bsz, blk, -1).transpose(0, 1)

class PlasticLinear( nn.Module):

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

class Linear( nn.Linear):

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