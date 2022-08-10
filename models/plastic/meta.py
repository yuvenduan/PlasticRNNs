__all__ = ['PlasticModule', 'PlasticParam']

import typing as tp
import torch
import numpy as np
import torch.nn as nn

class PlasticModule(nn.Module):

	def floatparams(self):
		paramlist = []
		lrlist = []
		for param in self.modules():
			if isinstance(param, PlasticParam):
				paramlist.append(param.floatparam)
				lrlist.append(param.lr)
		return paramlist, lrlist

	def set_floatparam(self, data):
		count = 0
		for param in self.modules():
			if isinstance(param, PlasticParam):
				d = param.total_dim
				param.set_floatparam(data[:, count: count + d])
				count += d

	def get_floatparam_dim(self) -> int:
		count = 0
		for param in self.modules():
			if isinstance(param, PlasticParam):
				d = param.total_dim
				count += d
		return count

	def update_floatparam(self, loss, lr, wd, max_norm) -> torch.Tensor:
		params, lrs = self.floatparams()
		grads = torch.autograd.grad(loss, params, create_graph=True)

		norm = 0
		for grad in grads:
			norm = norm + grad.square().sum()
		norm = norm.sqrt()
		if norm > max_norm:
			lr = lr * max_norm / norm

		new_param_list = []
		for grad, param, param_lr in zip(grads, params, lrs):
			param = (1 - wd) * param + lr * grad * param_lr
			new_param_list.append(param)

		new_param = torch.cat([param.view(param.shape[0], -1) for param in new_param_list], dim=1)
		return new_param

class PlasticParam(PlasticModule):

	lr_mode = 'uniform'

	@classmethod
	def set_elementwise_lr(cls, mode):
		if mode is not None:
			cls.lr_mode = mode

	def __init__(self, param: torch.Tensor):
		super().__init__()
		self.param = nn.Parameter(param)
		
		if self.lr_mode == 'none':
			self.lr = 1
		elif self.lr_mode == 'uniform':
			self.lr = nn.Parameter(torch.ones_like(param))
		elif self.lr_mode == 'random':
			self.lr = nn.Parameter(torch.rand_like(param) * 2 - 1)
		else:
			raise NotImplementedError(f"Unrecognized mode {self.lr_mode}")
		
		self.floatparam: tp.Optional[ torch.Tensor] = None
		self.total_dim = np.prod(self.param.shape)

	def forward(self) -> torch.Tensor:
		assert torch.is_grad_enabled(), "Gradient must be enabled"
		assert self.floatparam is not None, "Parameter not initialized"
		return self.floatparam + self.param

	def clear_floatparam(self):
		self.floatparam = None

	def set_floatparam(self, data: torch.Tensor):
		self.clear_floatparam()
		self.floatparam = data.view(-1, *self.param.shape)
		self.floatparam.requires_grad_(True)

	def __repr__(self):
		target = self.param
		return f'{self.__class__.__name__}{tuple( target.shape)}'