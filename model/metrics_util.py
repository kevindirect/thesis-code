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


# ********** PYTORCH LIGHTNING METRICS **********
class SimulatedReturn(Metric):
	"""
	Simulated Return pytorch lightning Metric.
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
		* potential maximum return

	Call self.reset() to clear the state lists and begin a new return period.
	"""
	def __init__(self, use_conf=True, use_kelly=False, threshold=.5,
		compute_on_step=False, dist_sync_on_step=False, process_group=None):
		"""
		use_conf (bool): whether to use confidence score for betsizing
		use_kelly (bool): whether to use kelly criterion for betsizing
		threshold (float): threshold to determine prediction direction
		"""
		super().__init__(compute_on_step=compute_on_step, \
			dist_sync_on_step=dist_sync_on_step, process_group=process_group)
		self.use_conf = use_conf
		self.use_kelly = use_kelly
		self.threshold = threshold
		self.name = {
			not self.use_conf: 'bret',			# simple binary return
			self.use_conf and not self.use_kelly: 'cret',	# conf betsize return
			self.use_conf and self.use_kelly: 'kret'	# kelly betsize return
		}.get(True)
		self.add_state('hit_dir', default=[], dist_reduce_fx='cat')
		self.add_state('reward', default=[], dist_reduce_fx='cat')
		if (self.use_conf):
			self.add_state('betsize', default=[], dist_reduce_fx='cat')

	def update(self, pred, actual, reward):
		"""
		Update simulated return state taking into account use_conf and use_kelly flags.
		Each call to update appends the new state tensors to the lists of current state
		tensors.

		Args:
			pred: prediction direction/confidence in interval [0, 1],
				below threshold is bearish, above threshold is bullish.
				The proximity to 0 or 1 represents confidence of the prediction.
			actual: actual direction or direction/confidence in interval [0, 1],
				can be a float or binary integer value.
			reward: value gained or lost based on correctness of prediction,
				e.g. the percent change over the period.

		Returns:
			None
		"""
		pred_dir = torch.sign(pred - self.threshold)
		actual_dir = torch.sign(actual - self.threshold)
		hit_dir = torch.ones_like(pred)			# reward hit
		hit_dir[pred_dir.eq(self.threshold)] = 0	# no trade, no reward
		hit_dir[pred_dir != actual_dir] = -1		# penalize misses
		self.hit_dir.append(hit_dir)
		self.reward.append(reward)

		if (self.use_conf):
			# confidence score / probability betsize
			betsize = torch.clamp(2 * abs(pred-self.threshold), min=0.0, max=1.0)
			if (self.use_kelly):
				# even money kelly
				betsize = torch.clamp(2 * betsize - 1, min=0.0, max=1.0)
			self.betsize.append(betsize)

	def compute(self, pfx=''):
		"""
		Compute return stats using hit dir (whether prediction hit or not),
		betsize (size of bet in [0, 1]), and the reward.
		Does not use the confidence score to scale return if use_conf is False.
		"""
		hit_dir = torch.cat(self.hit_dir, dim=0)
		reward = torch.abs(torch.cat(self.reward, dim=0))
		returns = hit_dir * reward
		if (self.use_conf):
			betsize = torch.cat(self.betsize, dim=0)
			returns *= betsize
		returns_cumsum = returns.cumsum(dim=0)

		return {
			f'{pfx}_{self.name}': returns_cumsum[-1],
			f'{pfx}_{self.name}_min': returns_cumsum.min(),
			f'{pfx}_{self.name}_max': returns_cumsum.max(),
			f'{pfx}_{self.name}_sharpe': returns.mean()/returns.std(),
			f'{pfx}_{self.name}_high': reward.sum()
		}


# """ ********** KERAS METRICS ********** """
# def mean_pred(y_true, y_pred):
# 	return K.mean(y_pred)

# def f1(y_true, y_pred):

# 	# Count positive samples.
# 	c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
# 	c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
# 	c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

# 	# If there are no true samples, fix the F1 score at 0.
# 	if c3 == 0:
# 		return 0

# 	# How many selected items are relevant?
# 	precision = c1 / c2

# 	# How many relevant items are selected?
# 	recall = c1 / c3

# 	# Calculate f1_score
# 	f1_score = 2 * (precision * recall) / (precision + recall)
# 	return f1_score

#  def mcor(y_true, y_pred):
# 	# matthews_correlation
# 	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
# 	y_pred_neg = 1 - y_pred_pos

# 	y_pos = K.round(K.clip(y_true, 0, 1))
# 	y_neg = 1 - y_pos

# 	tp = K.sum(y_pos * y_pred_pos)
# 	tn = K.sum(y_neg * y_pred_neg)

# 	fp = K.sum(y_neg * y_pred_pos)
# 	fn = K.sum(y_pos * y_pred_neg)

# 	numerator = (tp * tn - fp * fn)
# 	denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

# 	return numerator / (denominator + K.epsilon())

# def precision(y_true, y_pred):
# 	"""Precision metric.

# 	Only computes a batch-wise average of precision.

# 	Computes the precision, a metric for multi-label classification of
# 	how many selected items are relevant.
# 	"""
# 	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
# 	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
# 	precision = true_positives / (predicted_positives + K.epsilon())
# 	return precision

# def recall(y_true, y_pred):
# 	"""Recall metric.

# 	Only computes a batch-wise average of recall.

# 	Computes the recall, a metric for multi-label classification of
# 	how many relevant items are selected.
# 	"""
# 	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
# 	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
# 	recall = true_positives / (possible_positives + K.epsilon())
# 	return recall

# def f1(y_true, y_pred):
# 	def recall(y_true, y_pred):
# 		"""Recall metric.

# 		Only computes a batch-wise average of recall.

# 		Computes the recall, a metric for multi-label classification of
# 		how many relevant items are selected.
# 		"""
# 		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
# 		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
# 		recall = true_positives / (possible_positives + K.epsilon())
# 		return recall

# 	def precision(y_true, y_pred):
# 		"""Precision metric.

# 		Only computes a batch-wise average of precision.

# 		Computes the precision, a metric for multi-label classification of
# 		how many selected items are relevant.
# 		"""
# 		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
# 		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
# 		precision = true_positives / (predicted_positives + K.epsilon())
# 		return precision
# 	precision = precision(y_true, y_pred)
# 	recall = recall(y_true, y_pred)
# 	return 2*((precision*recall)/(precision+recall+K.epsilon()))
