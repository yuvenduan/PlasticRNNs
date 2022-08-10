__all__ = ['PlasticRNNCell', 'PlasticLSTMCell', 'VanillaRNNCell']

from .linear import *
from .activation import *
from .norm import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PlasticRNNCell(nn.Module):

    def __init__(self, ind, outd):
        super().__init__()
        # self.module = nn.ModuleList([nn.RNNCell(ind, outd, nonlinearity='relu') for i in range(num_block)])
        self.h_fc = PlasticLinear(outd, outd, scale=1)
        self.i_fc = PlasticLinear(ind, outd, scale=1)

    def forward(self, x: torch.Tensor, hx: torch.Tensor):
        x = F.relu(self.h_fc(hx) + self.i_fc(x))
        return x
    
class PlasticLSTMCell(nn.Module):

    def __init__(self, ind, outd):
        super().__init__()
        scale = 2
        self.h_fc = PlasticLinear(outd, outd * 4, scale=scale)
        self.i_fc = PlasticLinear(ind, outd * 4, scale=scale)

    def forward(self, x, hidden):
        hx, cx = hidden
        gates = self.h_fc(hx) + self.i_fc(x)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=-1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        return hy, cy

class VanillaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity="tanh", ct=False):
        super(VanillaRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.ct = ct

        self.weight_ih = nn.Parameter(torch.zeros((input_size, hidden_size)))
        self.weight_hh = nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()

        if self.nonlinearity == "tanh":
            self.act = torch.tanh
        elif self.nonlinearity == "relu":
            self.act = F.relu
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inp, hidden_in):
        if not self.ct:
            hidden_out = self.act(torch.matmul(inp, self.weight_ih)
                                  + torch.matmul(hidden_in, self.weight_hh)
                                  + self.bias)
        else:
            alpha = 0.1
            hidden_out = (1 - alpha) * hidden_in \
                         + alpha * self.act(torch.matmul(inp, self.weight_ih)
                                            + torch.matmul(hidden_in, self.weight_hh)
                                            + self.bias)
        return hidden_out