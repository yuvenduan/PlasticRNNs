__all__ = ['PlasticModule', 'PlasticParam']

import typing as tp
import torch
import numpy as np
import torch.nn as nn

class PlasticParam(nn.Module):

	lr_mode = 'uniform'
	requires_param_grad = True

	@classmethod
	def set_elementwise_lr(cls, mode):
		if mode is not None:
			cls.lr_mode = mode

	@classmethod
	def set_param_grad(cls, mode=True):
		cls.requires_param_grad = mode

	def __init__(self, param: torch.Tensor):
		super().__init__()
		self.param = nn.Parameter(param, requires_grad=self.requires_param_grad)
		
		# initialize connection-wise learning rates according to the configuration
		if self.lr_mode == 'none':
			self.lr = 1
		elif self.lr_mode == 'uniform':
			self.lr = nn.Parameter(torch.ones_like(param))
		elif self.lr_mode == 'neg_uniform':
			self.lr = nn.Parameter(torch.full_like(param, -1))
		elif self.lr_mode == 'random':
			self.lr = nn.Parameter(torch.rand_like(param) * 2 - 1)
		else:
			raise NotImplementedError(f"Unrecognized mode {self.lr_mode}")
		
		self.floatparam: tp.Optional[ torch.Tensor] = None
		self.pre: tp.Optional[ torch.Tensor] = None
		self.post: tp.Optional[ torch.Tensor] = None
		self.total_dim = np.prod(self.param.shape)

	def forward(self) -> torch.Tensor:
		# assert torch.is_grad_enabled(), "Gradient must be enabled"
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

class PlasticModule(nn.Module):

	def floatparams(self) -> list[PlasticParam]:
		paramlist = []
		for param in self.modules():
			if isinstance(param, PlasticParam):
				paramlist.append(param)
		return paramlist

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

	def update_floatparam(self, loss, lr, wd, max_norm, mode='gradient') -> torch.Tensor:
		params = self.floatparams()

		# calculate how params change at the current time step
		if mode == 'gradient':
			floatparams = [param.floatparam for param in params]
			grads = torch.autograd.grad(loss, floatparams, create_graph=True)

		elif mode == 'hebbian':
			grads = []
			for param in params:
				if param.param.dim() == 2:
					grad = torch.bmm(param.pre.unsqueeze(-1), param.post.unsqueeze(-2))
				else:
					grad = torch.zeros_like(param.floatparam)
				grads.append(grad)

		else:
			raise NotImplementedError(mode)

		# shrink the learning rate according the norm
		norm = 0
		for grad in grads:
			norm = norm + grad.square().sum(dim=tuple(range(1, grad.dim())))
		norm = norm.sqrt()
		lr = lr - lr * (1 - max_norm / norm) * (norm > max_norm)

		lrs = [lr, ]
		wds = [wd, ]
		for d in range(2):
			lrs.append(lrs[-1].unsqueeze(-1))
			wds.append(wds[-1].unsqueeze(-1))

		# calculate new params
		new_param_list = []
		for grad, param in zip(grads, params):
			new_param = (1 - wds[param.param.dim()]) * param.floatparam + lrs[param.param.dim()] * grad * param.lr
			new_param_list.append(new_param)

		new_param = torch.cat([param.view(param.shape[0], -1) for param in new_param_list], dim=1)
		return new_param