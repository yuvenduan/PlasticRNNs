
__all__ = ['PlasticLayerNorm']

from ..plastic import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class PlasticLayerNorm( nn.Module):

    def __init__(self, normalized_shape, eps = 1e-5):
        super().__init__()
        self.weight = PlasticParam(torch.ones(normalized_shape))
        self.bias = PlasticParam(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight(), self.bias(), self.eps)