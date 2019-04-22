"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
from hyperopt import hp, STATUS_OK

from common_util import MODEL_DIR
from model.common import PYTORCH_MODELS_DIR, ERROR_CODE, TEST_RATIO, VAL_RATIO
from model.model_p.classifier import Classifier
from model.model_p.regressor import Regressor
from model.model_p.temporal_mixin import TemporalMixin
from model.model_p.tcn_util import TCN


class TCN_CLF(TemporalMixin, Classifier):
	"""
	Top level Temporal CNN Classifer.
	Note: Receptive Field Size = Number TCN Blocks * Kernel Size * Last Layer Dilation Size

	Parameters:
		input_windows (int > 0): Number of aggregation windows in the input layer
		topology (list): Topology of the TCN divided by the window size
		kernel_size (int > 1): CNN kernel size
		dropout (float [0, 1]): dropout probability, probability of an element to be zeroed
		attention (bool): whether or not to include attention block after each tcn block
		max_attn_len (int > 0): max length of attention (only relevant if attention is set to True)
	"""
	def __init__(self, other_space={}):
		default_space = {
			'input_windows': hp.uniform('input_windows', 3, 20),
			'topology': hp.choice('topology', [[1, 1], [1, 3], [1, 5], [1, 7],
												[3, 1], [3, 3], [3, 5], [3, 7],
												[5, 1], [5, 3], [5, 5], [5, 7],
												[7, 1], [7, 3], [7, 5], [7, 7]]),
			'kernel_size': hp.uniform('kernel_size', 2, 10),
			'dropout': hp.uniform('dropout', .01, .80),
			'attention': hp.choice('attention', [False]),
			'max_attn_len': hp.uniform('max_attn_len', 24, 120)
		}
		super(TCN_CLF, self).__init__({**default_space, **other_space})

	def make_model(self, params, input_shape, num_outputs=2):
		window_size = input_shape[1]
		eff_history = window_size * params['input_windows']  								# Effective history = window_size * input_windows
		real_topology = window_size * np.array(params['topology'])							# Scale topology by the window size
		real_topology = np.clip(real_topology, a_min=1, a_max=None).astype(int)				# Make sure that layer outputs are always greater than zero
		mdl = TCN(num_input_channels=input_shape[0], channels=real_topology.tolist(), num_outputs=num_outputs, kernel_size=params['kernel_size'],
							dropout=params['dropout'], attention=params['attention'], max_attn_len=params['max_attn_len'])
		return mdl

class TCN_REG(TemporalMixin, Regressor):
	"""
	Top level Temporal CNN Regressor.
	Note: Receptive Field Size = Number TCN Blocks * Kernel Size * Last Layer Dilation Size

	Parameters:
		input_windows (int > 0): Number of aggregation windows in the input layer
		topology (list): Topology of the TCN divided by the window size
		kernel_size (int > 1): CNN kernel size
		dropout (float [0, 1]): dropout probability, probability of an element to be zeroed
		attention (bool): whether or not to include attention block after each tcn block
		max_attn_len (int > 0): max length of attention (only relevant if attention is set to True)
	"""
	def __init__(self, other_space={}):
		default_space = {
			'input_windows': hp.uniform('input_windows', 3, 20),
			'topology': hp.choice('topology', [[1, 1], [1, 3], [1, 5], [1, 7],
												[3, 1], [3, 3], [3, 5], [3, 7],
												[5, 1], [5, 3], [5, 5], [5, 7],
												[7, 1], [7, 3], [7, 5], [7, 7]]),
			'kernel_size': hp.uniform('kernel_size', 2, 10),
			'dropout': hp.uniform('dropout', .01, .80),
			'attention': hp.choice('attention', [False]),
			'max_attn_len': hp.uniform('max_attn_len', 24, 120)
		}
		super(TCN_REG, self).__init__({**default_space, **other_space})

	def make_model(self, params, input_shape, num_outputs=1):
		window_size = input_shape[1]
		eff_history = window_size * params['input_windows']  								# Effective history = window_size * input_windows
		real_topology = window_size * np.array(params['topology'])							# Scale topology by the window size
		real_topology = np.clip(real_topology, a_min=1, a_max=None).astype(int)				# Make sure that layer outputs are always greater than zero
		mdl = TCN(num_input_channels=input_shape[0], channels=real_topology.tolist(), num_outputs=num_outputs, kernel_size=params['kernel_size'],
							dropout=params['dropout'], attention=params['attention'], max_attn_len=params['max_attn_len'])
		return mdl
