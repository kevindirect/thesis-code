# Kevin Patel

import sys
import os
from os import sep
from os.path import splitext
from functools import partial, reduce
import logging

import numpy as np
import pandas as pd
from dask import delayed, compute, visualize
import keras.backend as K

from common_util import MODEL_DIR, JSON_SFX_LEN, DT_CAL_DAILY_FREQ, get_cmd_args, in_debug_mode, reindex_on_time_mask, gb_transpose, pd_common_index_rows, filter_cols_below, dump_df, load_json, outer_join, list_get_dict, chained_filter, benchmark
from model.common import DATASET_DIR, FILTERSET_DIR, EXPECTED_NUM_HOURS
from recon.model_util import get_train_test_split, gen_time_series_split



""" ********** KERAS METRICS ********** """
def mean_pred(y_true, y_pred):
	return K.mean(y_pred)

def f1(y_true, y_pred):

	# Count positive samples.
	c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
	c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

	# If there are no true samples, fix the F1 score at 0.
	if c3 == 0:
		return 0

	# How many selected items are relevant?
	precision = c1 / c2

	# How many relevant items are selected?
	recall = c1 / c3

	# Calculate f1_score
	f1_score = 2 * (precision * recall) / (precision + recall)
	return f1_score

 def mcor(y_true, y_pred):
	# matthews_correlation
	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
	y_pred_neg = 1 - y_pred_pos

	y_pos = K.round(K.clip(y_true, 0, 1))
	y_neg = 1 - y_pos

	tp = K.sum(y_pos * y_pred_pos)
	tn = K.sum(y_neg * y_pred_neg)

	fp = K.sum(y_neg * y_pred_pos)
	fn = K.sum(y_pos * y_pred_neg)

	numerator = (tp * tn - fp * fn)
	denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

	return numerator / (denominator + K.epsilon())

def precision(y_true, y_pred):
	"""Precision metric.

	Only computes a batch-wise average of precision.

	Computes the precision, a metric for multi-label classification of
	how many selected items are relevant.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def recall(y_true, y_pred):
	"""Recall metric.

	Only computes a batch-wise average of recall.

	Computes the recall, a metric for multi-label classification of
	how many relevant items are selected.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def f1(y_true, y_pred):
	def recall(y_true, y_pred):
		"""Recall metric.

		Only computes a batch-wise average of recall.

		Computes the recall, a metric for multi-label classification of
		how many relevant items are selected.
		"""
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall

	def precision(y_true, y_pred):
		"""Precision metric.

		Only computes a batch-wise average of precision.

		Computes the precision, a metric for multi-label classification of
		how many selected items are relevant.
		"""
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		return precision
	precision = precision(y_true, y_pred)
	recall = recall(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))
