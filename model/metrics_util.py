# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.metrics.metric import TensorMetric
# import keras.backend as K

from common_util import MODEL_DIR, isnt
from model.common import PYTORCH_LOSS_MAPPING 


# ********** PYTORCH LIGHTNING METRICS **********
class SimulatedReturn(TensorMetric):
	"""
	Simulated Return pytorch lightning TensorMetric.
	Maximum trade value of $1.00 per trade.
	Assumes zero transaction cost.
	"""
	def __init__(self, return_type='binary_confidence', reduce_group=None, reduce_op=None):
		super().__init__(type(self).__name__, reduce_group, reduce_op)
		self.return_fn = {
			'binary': SimulatedReturn.binary,
			'binary_confidence': SimulatedReturn.binary_confidence
		}.get(return_type)

	@classmethod
	def binary(cls, pred, actual, reward, threshold=.5):
		"""
		Calculate simulated binary return.
		Size of each bet is $1.

		Args:
			pred: prediction direction/confidence in interval [0, 1],
				below threshold is bearish, above threshold is bullish.
			actual: actual direction or direction/confidence in interval [0, 1],
				can be a float or binary integer value.
			reward: value gained or lost based on correctness of prediction,
				e.g. the percent change over the period.
			threshold: threshold where neutral (no direction) is, default is 0.5

		Returns:
			hit/miss * reward/penalty
		"""
		pred_dir = torch.sign(pred - threshold)
		actual_dir = torch.sign(actual - threshold)
		hit_dir = torch.ones_like(pred)			# reward hit
		hit_dir[pred_dir.eq(threshold)] = 0		# no trade, no reward
		hit_dir[pred_dir != actual_dir] = -1		# penalize misses
		return hit_dir * torch.abs(reward)

	@classmethod
	def binary_confidence(cls, pred, actual, reward, threshold=.5, kelly=False):
		"""
		Calculate simulated binary confidence return.
		Same as binary return except each $1 bet is scaled by a confidence
		value so that the final bet size is min(2 * abs(pred-threshold), 1).

		Args:
			pred: prediction direction/confidence in interval [0, 1],
				below threshold is bearish, above threshold is bullish.
				The proximity to 0 or 1 represents confidence of the prediction.
			actual: actual direction or direction/confidence in interval [0, 1],
				can be a float or binary integer value.
			reward: value gained or lost based on correctness of prediction,
				e.g. the percent change over the period.
			threshold: threshold where neutral (no direction) is, default is 0.5
			kelly: whether to use even money kelly criterion for bet sizing

		Returns:
			hit/miss * betsize * reward/penalty
		"""
		pred_dir = torch.sign(pred - threshold)
		actual_dir = torch.sign(actual - threshold)
		hit_dir = torch.ones_like(pred)			# reward hit
		hit_dir[pred_dir.eq(threshold)] = 0		# no trade, no reward
		hit_dir[pred_dir != actual_dir] = -1		# penalize misses
		betsize = torch.clamp(2 * abs(pred-threshold), min=0.0, max=1.0)	# confidence score / probability betsize
		if (kelly):
			betsize = torch.clamp(2 * betsize - 1, min=0.0, max=1.0)	# even money kelly
		return hit_dir * betsize * torch.abs(reward)

	def forward(self, pred, actual, reward, **kwargs):
		with torch.no_grad():
			return self.return_fn(pred, actual, reward, **kwargs)

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
