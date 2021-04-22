"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
import torch
from torchmetrics import Metric

from common_util import MODEL_DIR, isnt, pt_diff1d
from model.common import PYTORCH_LOSS_MAPPING


# ********** PYTORCH LIGHTNING RETURN UTILS **********
def get_cumulative_return(asset_returns, compounded=False):
	if (compounded):
		cum_returns = (asset_returns + 1.0).cumprod(dim=0) - 1.0
	else:
		cum_returns = asset_returns.cumsum(dim=0)
	return cum_returns

def get_sharpe_ratio(cum_returns, annualized=True):
	ret = pt_diff1d(cum_returns)
	ret_std = ret.std()
	ret_std = ret_std if (ret_std != 0) else 1
	sr = ret.mean() / ret_std
	if (annualized):
		sr *= np.sqrt(252).item(0)
	return sr


# ********** PYTORCH LIGHTNING RETURN BENCHMARKS **********
class BuyAndHoldStrategy(Metric):
	"""
	Buy and hold benchmark strategy.
	If the return series used consists of daily open-to-close returns,
	this will be buy and hold with no overnight risk (buy at open, sell at close daily).
	All bets are sized $1 per trade. Assumes zero transaction cost.
	"""
	def __init__(self, compounded=False, compute_on_step=False, dist_sync_on_step=False, process_group=None):
		super().__init__(compute_on_step=compute_on_step, \
			dist_sync_on_step=dist_sync_on_step, process_group=process_group)
		self.name = 'bs'
		self.compounded = compounded
		if (self.compounded):
			self.name += 'c'
		self.add_state('returns', default=[], dist_reduce_fx='cat')

	def update(self, pred, actual, actual_ret):
		self.returns.append(actual_ret)

	def compute(self, pfx=''):
		returns = torch.cat(self.returns, dim=0)
		cr = get_cumulative_return(returns, compounded=self.compounded)

		return {
			f'{pfx}_{self.name}': cr[-1],
			f'{pfx}_{self.name}_sharpe': get_sharpe_ratio(cr),
			f'{pfx}_{self.name}_min': cr.min(),
			f'{pfx}_{self.name}_max': cr.max()
		}


class OptimalStrategy(Metric):
	"""
	Optimal return benchmark strategy.
	This is as if an agent makes perfect predictions over the period using
	the maximum bet size ($1) in every trade - sum(abs(returns))
	All bets are sized $1 per trade. Assumes zero transaction cost.
	"""
	def __init__(self, compounded=False, compute_on_step=False, dist_sync_on_step=False, process_group=None):
		super().__init__(compute_on_step=compute_on_step, \
			dist_sync_on_step=dist_sync_on_step, process_group=process_group)
		self.name = 'os'
		self.compounded = compounded
		if (self.compounded):
			self.name += 'c'
		self.add_state('returns', default=[], dist_reduce_fx='cat')

	def update(self, pred, actual, actual_ret):
		self.returns.append(torch.abs(actual_ret))

	def compute(self, pfx=''):
		returns = torch.cat(self.returns, dim=0)
		cr = get_cumulative_return(returns, compounded=self.compounded)

		return {
			f'{pfx}_{self.name}': cr[-1],
			f'{pfx}_{self.name}_sharpe': get_sharpe_ratio(cr)
		}


# ********** PYTORCH LIGHTNING RETURN METRICS **********
class SimulatedReturn(Metric):
	"""
	Simulated Return Pytorch Lightning Metric.
	Starting maximum bet size of $1 per trade. Assumes zero transaction cost.

	Uses confidence score based betsizing if use_conf is True, this allows
	the betsize to be less than the max.
	If compounded is true returns are compounded, otherwise the max size of each
	return is $1.
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
	def __init__(self, use_conf=True, use_kelly=False, compounded=False, pred_type='reg',
		compute_on_step=False, dist_sync_on_step=False, process_group=None):
		"""
		use_conf (bool): whether to use confidence score for betsizing
		use_kelly (bool): whether to use kelly criterion for betsizing
		threshold (float): threshold to determine prediction direction
		pred_type ('clf'|'reg'):
		"""
		super().__init__(compute_on_step=compute_on_step, \
			dist_sync_on_step=dist_sync_on_step, process_group=process_group)
		if (use_kelly):
			assert use_conf, 'must set use_conf to use kelly'
		self.use_conf = use_conf
		self.use_kelly = use_kelly
		self.compounded = compounded
		self.name = {
			not self.use_conf: 'br',			# simple binary return
			self.use_conf and not self.use_kelly: 'cr',	# conf betsize return
			self.use_kelly: 'kr'				# kelly betsize return
		}.get(True)
		if (self.compounded):
			self.name += 'c'
		self.pred_type = pred_type
		self.add_state('pred_dir', default=[], dist_reduce_fx='cat')
		self.add_state('actual_ret', default=[], dist_reduce_fx='cat')
		if (self.use_conf):
			self.add_state('betsize', default=[], dist_reduce_fx='cat')
		else:
			self.betsize = 1

	def update(self, pred, actual_ret):
		"""
		Update simulated return state taking into account use_conf and use_kelly flags.
		Each call to update appends the new state tensors to the lists of current state
		tensors.

		Args:
			pred: predictions matrix
			actual_ret: returns matrix

		Returns:
			None
		"""
		# TODO include uncertainty estimate in simulated trading return

		if (self.pred_type == 'clf'): # expects prediction in [-1, 1]
			pred_conf, pred_dir = torch.abs(pred), torch.sign(pred)
			self.pred_dir.append(pred_dir)
			self.actual_ret.append(actual_ret)

			if (self.use_conf):
				if (self.use_kelly):
					pred_conf = torch.clamp(2 * pred_conf - 1, min=0.0, max=1.0) # even money kelly
				self.betsize.append(pred_conf)
		elif (self.pred_type == 'reg'): # expects a floating point return prediction
			pred_dir = torch.sign(pred)
			self.pred_dir.append(pred_dir)
			self.actual_ret.append(actual_ret)
			pred_conf = 1 # XXX keep it simple for now

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
		cr = get_cumulative_return(returns, compounded=self.compounded)

		return {
			f'{pfx}_{self.name}': cr[-1],
			f'{pfx}_{self.name}_sharpe': get_sharpe_ratio(cr),
			f'{pfx}_{self.name}_min': cr.min(),
			f'{pfx}_{self.name}_max': cr.max()
		}

