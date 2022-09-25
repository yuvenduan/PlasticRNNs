__all__ = ['Regression']

import torch
import torch.utils.data as tud
import torch.nn as nn

from configs.configs import RegressionConfig

class Regression(tud.Dataset):

    def __init__(self, config: RegressionConfig):

        self.n_var = config.input_shape[0] - 2
        self.input_noise = config.input_noise
        self.train_length = config.train_length
        self.test_length = config.test_length
        self.task_mode = config.task_mode

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        
        if self.task_mode == 'linear':
            f = nn.Linear(self.n_var, 1)
        elif self.task_mode == 'mlp':
            f = nn.Sequential(
                nn.Linear(self.n_var, 2),
                nn.Tanh(),
                nn.Linear(2, 1)
            )
        else:
            raise NotImplementedError(self.task_mode)
        f.requires_grad_(False)

        variables = torch.rand((self.train_length + self.test_length, self.n_var, )) * 2 - 1
        values: torch.Tensor = f(variables)
        values = (values - values.mean()) / values.std()
        
        input = torch.zeros((self.train_length + self.test_length, self.n_var + 2))
        input[:, : self.n_var] = variables
        input[: self.train_length, -2: -1] = values[: self.train_length] + \
            torch.randn_like(values[: self.train_length]) * self.input_noise
        input[: self.train_length, -1: ] = 1

        return input, values