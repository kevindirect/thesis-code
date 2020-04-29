"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd

from common_util import MODEL_DIR
from model.common import PYTORCH_MODELS_DIR, ERROR_CODE, TEST_RATIO, VAL_RATIO
from model.models.mixins.temporal_mixin import TemporalMixin
from model.models.tcn_util import TCN


def tcn_fix_params(params):
	"""
	Fix parameters sampled from hyperopt space dictionary.
	"""
	params['epochs'] = int(params['epochs'])
	params['batch_size'] = int(params['batch_size'])
	params['input_windows'] = int(params['input_windows'])
	params['kernel_size'] = int(params['kernel_size'])
	params['max_attn_len'] = int(params['max_attn_len'])


class TCN_CLF(TemporalMixin):
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
	def fix_params(self, params):
		tcn_fix_params(params)

	def make_model(self, params, obs_shape, num_outputs=2):
		# TODO - make this a static method?
		input_channels, window_size = obs_shape
		eff_history = window_size * params['input_windows']					# Effective history = window_size * input_windows
		real_topology = window_size * np.array(params['topology'])				# Scale topology by the window size
		real_topology = np.clip(real_topology, a_min=1, a_max=None).astype(int)			# Make sure that layer outputs are always greater than zero
		mdl = TCN(num_input_channels=input_channels, channels=real_topology.tolist(), num_outputs=num_outputs, kernel_size=params['kernel_size'],
							dropout=params['dropout'], attention=params['attention'], max_attn_len=params['max_attn_len'])
		return mdl

