"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.metrics import Metric

from common_util import MODEL_DIR, isnt
from model.common import PYTORCH_LOSS_MAPPING


# ********** PYTORCH LIGHTNING RETURN BENCHMARKS **********
class BuyAndHoldStrategy(Metric):
	"""
	Buy and hold benchmark strategy.
	If the return series used consists of daily open-to-close returns,
	this will be buy and hold with no overnight risk (buy at open, sell at close daily).
	All bets are sized $1 per trade. Assumes zero transaction cost.
	"""
	def __init__(self, compute_on_step=False, dist_sync_on_step=False, process_group=None):
		super().__init__(compute_on_step=compute_on_step, \
			dist_sync_on_step=dist_sync_on_step, process_group=process_group)
		self.name = 'bs'
		self.add_state('returns', default=[], dist_reduce_fx='cat')

	def update(self, pred, actual, actual_ret):
		self.returns.append(actual_ret)

	def compute(self, pfx=''):
		returns = torch.cat(self.returns, dim=0)
		returns_cumsum = returns.cumsum(dim=0)

		return {
			f'{pfx}_{self.name}': returns_cumsum[-1],
			f'{pfx}_{self.name}_sharpe': returns.mean()/returns.std(),
			f'{pfx}_{self.name}_min': returns_cumsum.min(),
			f'{pfx}_{self.name}_max': returns_cumsum.max(),
		}


class OptimalStrategy(Metric):
	"""
	Optimal return benchmark strategy.
	This is as if an agent makes perfect predictions over the period using
	the maximum bet size ($1) in every trade - sum(abs(returns))
	All bets are sized $1 per trade. Assumes zero transaction cost.
	"""
	def __init__(self, compute_on_step=False, dist_sync_on_step=False, process_group=None):
		super().__init__(compute_on_step=compute_on_step, \
			dist_sync_on_step=dist_sync_on_step, process_group=process_group)
		self.name = 'os'
		self.add_state('returns', default=[], dist_reduce_fx='cat')

	def update(self, pred, actual, actual_ret):
		self.returns.append(torch.abs(actual_ret))

	def compute(self, pfx=''):
		returns = torch.cat(self.returns, dim=0)
		return {
			f'{pfx}_{self.name}': returns.sum(),
			f'{pfx}_{self.name}_sharpe': returns.mean()/returns.std(),
		}


# ********** PYTORCH LIGHTNING RETURN METRICS **********
class SimulatedReturn(Metric):
	"""
	Simulated Return Pytorch Lightning Metric.
	Maximum bet size of $1 per trade. Assumes zero transaction cost.

	Uses confidence score based betsizing if use_conf is True, this allows
	the betsize to be less than $1.
	If use_conf and use_kelly are True, uses even kelly criterion based betsizing.

	The update method updates the class state (hit_dir, reward, and betsize).
	Each state list looks something like this:
		[tensor(a, b, c), tensor(d, e, f), tensor(g, h, i)]

	Each of these lists are flattened into tensors in the compute method and are used
	to compute simulated return metrics. These include:
		* end of period cumulative return
		* minimum of cumulative return (max drawdown)
		* maximum of cumulative return
		* sharpe ratio over period
	"""
	def __init__(self, use_conf=True, use_kelly=False,
		compute_on_step=False, dist_sync_on_step=False, process_group=None):
		"""
		use_conf (bool): whether to use confidence score for betsizing
		use_kelly (bool): whether to use kelly criterion for betsizing
		threshold (float): threshold to determine prediction direction
		"""
		super().__init__(compute_on_step=compute_on_step, \
			dist_sync_on_step=dist_sync_on_step, process_group=process_group)
		if (use_kelly):
			assert use_conf, 'must set use_conf to use kelly'
		self.use_conf = use_conf
		self.use_kelly = use_kelly
		self.name = {
			not self.use_conf: 'br',			# simple binary return
			self.use_conf and not self.use_kelly: 'cr',	# conf betsize return
			self.use_kelly: 'kr'				# kelly betsize return
		}.get(True)
		self.add_state('pred_dir', default=[], dist_reduce_fx='cat')
		self.add_state('actual_ret', default=[], dist_reduce_fx='cat')
		if (self.use_conf):
			self.add_state('betsize', default=[], dist_reduce_fx='cat')
		else:
			self.betsize = 1

	def update(self, pred, actual_dir, actual_ret):
		"""
		Update simulated return state taking into account use_conf and use_kelly flags.
		Each call to update appends the new state tensors to the lists of current state
		tensors.

		Args:
			pred: predictions matrix
			actual_dir: direction vector in interval [0, 1]
			actual_ret: returns matrix

		Returns:
			None
		"""
		pred_conf, pred_dir = pred.max(dim=-1)
		pred_dir[pred_dir==0] = -1
		aret = torch.gather(actual_ret, dim=-1, index=actual_dir.unsqueeze(-1)).squeeze()
		self.pred_dir.append(pred_dir)
		self.actual_ret.append(aret)

		if (self.use_conf):
			if (self.use_kelly):
				pred_conf = torch.clamp(2 * pred_conf - 1, min=0.0, max=1.0) # even money kelly
			self.betsize.append(pred_conf)

	def compute(self, pfx='', eps=10**-6):
		"""
		Compute return stats using pred_dir, actual return, and betsize.
		Does not use the confidence score to scale return if use_conf is False.
		TODO bull and bear only returns
		"""
		pred_dirs = torch.cat(self.pred_dir, dim=0)
		actual_rets = torch.cat(self.actual_ret, dim=0)
		betsizes = torch.cat(self.betsize, dim=0) if (self.use_conf) else self.betsize
		returns = pred_dirs * actual_rets * betsizes
		returns_cumsum = returns.cumsum(dim=0)

		return {
			f'{pfx}_{self.name}': returns_cumsum[-1],
			f'{pfx}_{self.name}_sharpe': returns.mean()/(returns.std()+eps),
			f'{pfx}_{self.name}_min': returns_cumsum.min(),
			f'{pfx}_{self.name}_max': returns_cumsum.max(),
		}

