"""
Kevin Patel
"""
import sys
import os
import logging
import pickle

import numpy as np
import pandas as pd
from scipy.stats import skew
import torch as pt
from torch.nn.functional import sigmoid
from torchmetrics.functional import accuracy, precision, recall, f1_score
from torchmetrics import Metric
import matplotlib.pyplot as plt

from common_util import MODEL_DIR, DT_FMT_YMD, is_valid, isnt, is_type, rectify_json, dump_json
from model.common import PYTORCH_LOSS_MAPPING, TRADING_DAYS
from recon.viz import plot_df_line_subplot, plot_df_scatter_subplot, plot_df_hist_subplot


# ********** PYTORCH LIGHTNING RETURN UTILS **********
def get_label_dist(f, l, t):
	res = pd.Series(l, dtype=int).value_counts(normalize=True).to_dict()
	return 'label_dist', res

def get_cumulative_profit(ret, compounded=False):
	"""
	Convert independent period returns to cumulative profit, with optional compounding.
	"""
	if (compounded):
		cum_profit = (ret + 1.0).cumprod(dim=0) - 1.0
	else:
		cum_profit = ret.cumsum(dim=0)
	return cum_profit

def get_cagr(cum_profit):
	"""
	Computes compund annual growth rate over daily returns.
	"""
	num_years = len(cum_profit) / TRADING_DAYS
	end_balance = cum_profit[-1] + 1.0 # assume starting balance of $1
	cagr = (end_balance**(1.0/num_years)) - 1.0
	return cagr

def get_sharpe_ratio(ret, rfr=0, annualized=True):
	"""
	Computes sharpe ratio over independent period returns.
	Computes information ratio if risk free rate is zero (default).
	"""
	ret_std = ret.std()
	ret_std = ret_std if (ret_std != 0) else 1
	sr = (ret.mean() - rfr) / ret_std
	if (annualized):
		sr *= np.sqrt(TRADING_DAYS).item(0)
	return sr

def get_skew(ret):
	"""
	Computes sample skew (third moment) of return distribution.
	"""
	return skew(ret)


# ********** PLOTTING UTILS **********
def plot_single(ret_df, profit_df, split, name, hist_bins=80):
	fig, axes = plt.subplots(3, 1, sharex=False, figsize=(25, 25))

	plot_df_line_subplot(profit_df, axes[0],
		title=f'{split} {name} Cumulative Non-Compounded Profit and Loss',
		ylabel='Cumulative P&L', colors='k')
	plot_df_scatter_subplot(ret_df, axes[1],
		title=f'{split} {name} Returns',
		xlabel=f'{split} Examples', ylabel='Return', colors='k')
	plot_df_hist_subplot(ret_df, axes[2],
		title=f'{split} {name} Distribution',
		xlabel=f'{split} Returns', ylabel='Frequency', colors='k', hist_bins=hist_bins)
	return fig, axes

def plot_three(ret_df, profit_df, split, name, hist_bins=20):
	fig, axes = plt.subplots(3, 1, sharex=False, figsize=(25, 25))

	plot_df_line_subplot(profit_df, axes[0],
		title=f'{split} {name} Cumulative Non-Compounded Profit and Loss',
		ylabel='Cumulative P&L', linestyles=['dashed', 'dotted', 'dashdot'])
	plot_df_scatter_subplot(ret_df, axes[1],
		title=f'{split} {name} Returns',
		xlabel=f'{split} Examples', ylabel='Return', alpha=.5, markers=['o', 'o', '.'])
	plot_df_hist_subplot(ret_df, axes[2],
		title=f'{split} {name} Distribution',
		xlabel=f'{split} Return', ylabel='Frequency', alpha=.5, hist_bins=hist_bins)
	return fig, axes


# ********** PYTORCH LIGHTNING RETURN BENCHMARKS **********
class ReturnMetric(Metric):
	"""
	Base Return torchmetrics.Metric class.
	Default parameters will compute a long buy and hold benchmark strategy.
	"""
	def __init__(self, name, use_dir=False, use_conf=False, compounded=False, \
		compute_on_step=False, dist_sync_on_step=False, process_group=None):
		super().__init__(compute_on_step=compute_on_step, \
			dist_sync_on_step=dist_sync_on_step, process_group=process_group)
		self.name = name
		self.use_dir = use_dir
		self.use_conf = use_conf
		self.compounded = compounded
		if (self.compounded):
			self.name += '_comp'
		self.plot_go_short = False

		self.add_state('actual_ret', default=[], dist_reduce_fx='cat')
		if (self.use_dir):
			self.add_state('pred_dir', default=[], dist_reduce_fx='cat')
		if (self.use_conf):
			self.add_state('bet_size', default=[], dist_reduce_fx='cat')

	def update(self, actual_ret):
		"""
		For classes that extend ReturnMetric, this is where the magic happens.
		"""
		self.actual_ret.append(actual_ret)

	def compute_pred_dirs(self, go_long, go_short):
		if (self.use_dir):
			assert go_long or go_short
			pred_dirs = pt.cat(self.pred_dir, dim=0)
			if (not go_long):
				pred_dirs[pred_dirs == 1] = 0
			if (not go_short):
				pred_dirs[pred_dirs == -1] = 0
		else:
			assert go_long != go_short
			pred_dirs = 1 if (go_long) else -1
		return pred_dirs

	def compute_bet_sizes(self):
		if (self.use_conf):
			bet_sizes = pt.cat(self.bet_size, dim=0)
		else:
			bet_sizes = 1
		return bet_sizes

	def compute_returns(self, go_long=True, go_short=True):
		actual_rets = pt.cat(self.actual_ret, dim=0)
		pred_dirs = self.compute_pred_dirs(go_long=go_long, go_short=go_short)
		bet_sizes = self.compute_bet_sizes()
		if (all(pt.isnan(ret := actual_rets * pred_dirs * bet_sizes))):
			ret = pt.zeros_like(ret)
		return ret

	def compute(self, prefix, go_long=True, go_short=True):
		"""
		Compute final stats.
		"""
		ret = self.compute_returns(go_long=go_long, go_short=go_short)
		profit = get_cumulative_profit(ret, self.compounded)
		avg_bet_size = self.compute_bet_sizes().mean() if (self.use_conf) else 1
		if (self.use_dir):
			preds = self.compute_pred_dirs(go_long, go_short)
			long_freq = len(preds[preds==1]) / len(preds)
		else:
			long_freq = 1 if (go_long) else 0

		return {
			f'{prefix}_longfreq': long_freq,
			f'{prefix}_avgbet': avg_bet_size,
			f'{prefix}_min': profit.min(), # max profit drawdown
			f'{prefix}_max': profit.max(), # max account profit
			f'{prefix}_profit': profit[-1],
			f'{prefix}_sharpe': get_sharpe_ratio(ret),
			f'{prefix}_skew': get_skew(ret),
			f'{prefix}_cagr': get_cagr(get_cumulative_profit(ret, compounded=True)),
		}

	def get_result_series(self, go_long=True, go_short=False):
		ret = {}
		if (go_long):
			ret['long'] = self.compute_returns(go_long=True, go_short=False)
		if (go_short):
			ret['short'] = self.compute_returns(go_long=False, go_short=True)
		if (go_long and go_short):
			ret['long+short'] = self.compute_returns(go_long=True, go_short=True)
		profit = {k: get_cumulative_profit(r, self.compounded) for k,r in ret.items()}
		return ret, profit

	def plot_result_series(self, split, name, index):
		ret, profit = self.get_result_series(go_long=True, go_short=self.plot_go_short)
		ret_df = pd.DataFrame(ret, index=index)
		profit_df = pd.DataFrame(profit, index=index)

		if (ret_df.shape[-1] == 1):
			fig, axes = plot_single(ret_df, profit_df, split, name)
		elif (ret_df.shape[-1] <= 3):
			fig, axes = plot_three(ret_df, profit_df, split, name)
		fig.tight_layout()
		return fig, axes

class BenchmarkHold(ReturnMetric):
	"""
	Hold benchmark strategy.
	If the return series used consists of daily open-to-close returns,
	this will be position hold with no overnight risk (open at market open, close at market close).
	All bets are sized $1 per trade. Assumes zero transaction cost.
	"""
	def __init__(self, compounded=False, compute_on_step=False, dist_sync_on_step=False, process_group=None):
		super().__init__('benchmark-hold', use_dir=False, use_conf=False, compounded=compounded,
			compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
			process_group=process_group)
		self.plot_go_short = False

	def get_clf_stats(self, prefix, num_classes=2):
		actual_dir = pt.sign(pt.cat(self.actual_ret, dim=0)).int()
		actual_dir[actual_dir==-1] = 0
		preds = pt.ones_like(actual_dir)
		acc = accuracy(preds, actual_dir)
		p = precision(preds, actual_dir, average='macro', num_classes=num_classes)
		r = recall(preds, actual_dir, average='macro', num_classes=num_classes)
		f1 = f1_score(preds, actual_dir, average='macro', num_classes=num_classes)
		return {
			f"{prefix}_clf_accuracy": acc,
			f"{prefix}_clf_precision": p,
			f"{prefix}_clf_recall": r,
			f"{prefix}_clf_f1": f1,
		}

	def compute(self, prefix, go_long=True, go_short=False):
		"""
		Compute final stats (sets default).
		"""
		return super().compute(prefix, go_long=go_long, go_short=go_short)

class BenchmarkOptimal(ReturnMetric):
	"""
	Optimal benchmark strategy.
	This is an agent making bets with perfect foresight over the period with
	the maximum bet size ($1) in every trade, simply "sum(abs(returns))"
	All bets are sized $1 per trade. Assumes zero transaction cost.
	"""
	def __init__(self, compounded=False, compute_on_step=False, dist_sync_on_step=False, process_group=None):
		super().__init__('benchmark-optimal', use_dir=True, use_conf=False, compounded=compounded,
			compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
			process_group=process_group)

	def get_clf_stats(self, prefix):
		return {
			f"{prefix}_clf_accuracy": 1,
			f"{prefix}_clf_precision": 1,
			f"{prefix}_clf_recall": 1,
			f"{prefix}_clf_f1": 1,
		}

	def update(self, actual_ret):
		self.actual_ret.append(actual_ret)
		self.pred_dir.append(pt.sign(actual_ret))

class BenchmarksMixin(object):
	"""
	Mixin for pl.DataModule.
	Uses data to produce benchmarks, over self.start:self.end if they exist
	in the host data module.
	"""
	def __init__(self):
		super().__init__()
		self.bench_fns = (BenchmarkHold, BenchmarkOptimal)
		self.stat_fns = (get_label_dist,)

	def get_benchmarks_split(self, split):
		"""
		Return results dict for a split, strat results are not computed.
		"""
		results = {'stat': {}}
		f, l, t = self.raw[split]
		idx = self.idx and self.idx[split]
		l = np.sum(l, axis=(1, 2), keepdims=False)

		if (is_valid(idx)):
			start = self.start and self.start[split]
			end = self.end and self.end[split]
			l = l[start:end]
			t = t[start:end]
		else:
			idx = self.raw_idx[split]
		t = pt.tensor(t[t!=0], dtype=pt.float32, requires_grad=False)

		results['stat'][f'date_start'] = idx[0].strftime(DT_FMT_YMD)
		results['stat'][f'date_end'] = idx[-1].strftime(DT_FMT_YMD)
		results['stat'][f'date_len'] = len(idx)

		for stat in self.stat_fns:
			res = stat(f, l, t)
			results['stat'][res[0]] = res[1]

		for strat in self.bench_fns:
			st = strat()
			st.update(t)
			results[st.name] = st

		return results

	def get_benchmarks(self):
		"""
		Return results dict for all splits, strat results are uncomputed.
		"""
		return {split: self.get_benchmarks_split(split) for split in self.raw.keys()}

	def compute_benchmarks_results_json(self, bench):
		"""
		Compute results dict for all splits.
		This function will modify the contents benchmark dictionary.
		"""
		for split in bench:
			for st_name in bench[split]:
				if (st_name == 'stat'): continue
				st = bench[split][st_name]
				res = st.get_clf_stats(split)
				res.update(st.compute(split))
				bench[split][st_name] = rectify_json(res)
		return bench

	def dump_benchmarks_results(self, bench, results_dir):
		results_json = self.compute_benchmarks_results_json(bench)
		for split, result in results_json.items():
			dump_json(result, split, results_dir)

	def dump_benchmarks_plots(self, bench, plot_dir):
		"""
		"""
		for split in bench:
			for st_name in bench[split]:
				if (st_name == 'stat'): continue
				st = bench[split][st_name]
				fig, axes = st.plot_result_series(split.title(),
					st_name.title(), self.idx[split])
				fname = f"{split}_{st_name}"

				with open(f'{plot_dir}{fname}.pickle', 'wb') as f:
					pickle.dump(fig, f)
				plt.savefig(f'{plot_dir}{fname}', bbox_inches="tight",
					transparent=True)
				plt.close(fig)


# ********** PYTORCH LIGHTNING RETURN METRICS **********
class SimulatedReturn(ReturnMetric):
	"""
	Simulated Return Pytorch Lightning Metric.
	Starting maximum bet size of $1 per trade. Assumes zero transaction cost.

	Uses confidence score based betsizing if use_conf is True, this allows
	the bet_size to be less than the max.
	If compounded is true returns are compounded, otherwise the max size of each
	return is $1.
	If use_conf and use_kelly are True, uses even kelly criterion based betsizing.

	The update method updates the class state (hit_dir, reward, and bet_size).
	Each state list looks something like this:
		[tensor(a, b, c), tensor(d, e, f), tensor(g, h, i)]

	Each of these lists are flattened into tensors in the compute method and are used
	to compute simulated return metrics. These include:
		* end of period cumulative return
		* minimum of cumulative return (max drawdown)
		* maximum of cumulative return
		* sharpe ratio over period
	"""
	def __init__(self, use_conf=True, use_kelly=False, compounded=False,
		pred_type='clf', dir_thresh=None, conf_thresh=None,
		compute_on_step=False, dist_sync_on_step=False, process_group=None):
		"""
		use_conf (bool): whether to use confidence score for betsizing
		use_kelly (bool): whether to use kelly criterion for betsizing
		dir_thresh (float>0|tuple(float<0,float>0)): threshold to determine
			prediction direction
		pred_type ('clf'|'reg'): 'clf' expects predictions in [-1, 1],
			'reg' expects floating point predictions
		"""
		self.use_conf = use_conf
		self.use_kelly = use_kelly
		name = {
			not self.use_conf: 'binary',
			self.use_conf and not self.use_kelly: 'conf',
			self.use_conf and self.use_kelly: 'kelly'
		}.get(True)
		super().__init__(name, use_dir=True, use_conf=use_conf, compounded=compounded,
			compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
			process_group=process_group)

		self.plot_go_short = False
		self.pred_type = pred_type
		self.dir_thresh = (-dir_thresh, dir_thresh) if (is_type(dir_thresh, float)) \
			else dir_thresh
		self.conf_thresh = (-conf_thresh, conf_thresh) if (is_type(conf_thresh, float)) \
			else conf_thresh
		assert isnt(self.dir_thresh) or self.dir_thresh[0] <= self.dir_thresh[1], \
			'short dir threshold must be less then or equal to long dir threshold'
		assert isnt(self.conf_thresh) or self.conf_thresh[0] <= self.conf_thresh[1], \
			'short conf threshold must be less then or equal to long conf threshold'

		if (is_valid(dt := self.dir_thresh) and is_valid(ct := self.conf_thresh) and dt==ct):
			self.name += f'_t({dt[1]:.3f})' if (-dt[0] == dt[1]) \
				else f'_t({dt[0]:.3f},{dt[1]:.3f})'
		else:
			if (is_valid(dt := self.dir_thresh)):
				self.name += f'_dt({dt[1]:.3f})' if (-dt[0] == dt[1]) \
					else f'_dt({dt[0]:.3f},{dt[1]:.3f})'
			if (is_valid(ct := self.dir_thresh)):
				self.name += f'_ct({ct[1]:.3f})' if (-ct[0] == ct[1]) \
					else f'_ct({ct[0]:.3f},{ct[1]:.3f})'

	def update(self, pred, actual_ret):
		"""
		Update simulated return state taking into account use_conf and use_kelly flags.
		Each call to update appends the new state tensors to the lists of current state
		tensors.
		# TODO uncertainty estimate in simulated trading return

		Args:
			pred: predictions matrix
			actual_ret: returns matrix

		Returns:
			None
		"""
		self.actual_ret.append(actual_ret)

		if (is_valid(dt := self.dir_thresh)):
			pred_dir = pt.zeros_like(pred)
			pred_dir[pred < dt[0]] = -1
			pred_dir[pred > dt[1]] = 1
		else:
			pred_dir = pt.sign(pred)
		self.pred_dir.append(pred_dir)

		if (self.use_conf):
			if (is_valid(ct := self.conf_thresh)):
				pred_conf = pt.zeros_like(pred)
				pred_conf[pred < ct[0]] = (pred[pred < ct[0]] - ct[0]) / (-1 - ct[0])
				pred_conf[pred > ct[1]] = (pred[pred > ct[1]] - ct[1]) / (1 - ct[1])
				# assert (pred_conf >= 0).all(), 'pred_conf must be in [0,1]'
			elif (self.pred_type == 'clf'):
				pred_conf = pred.abs()
			elif (self.pred_type == 'reg'):
				pred_conf = sigmoid(pred.abs())

			if (self.use_kelly): # even money kelly
				# pred_conf = pt.clamp(2 * pred_conf - 1, min=0.0, max=1.0)
				pred_conf = (2 * pred_conf - 1).to(pt.float32)\
					.clamp(min=0.0, max=1.0)
			self.bet_size.append(pred_conf)

