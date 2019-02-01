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
from model.model_p.binary_classifier import BinaryClassifier
from model.model_p.temporal_mixin import TemporalMixin
from model.model_p.cnn_util import CNN_Classifier


class BinaryCNN(TemporalMixin, BinaryClassifier):
	"""
	Top level CNN Binary Classifer.

	Parameters:
		input_windows (int > 0): Number of aggregation windows in the input layer
		topology (list): Topology of the CNN divided by the window size
		kernel_size (int > 1): CNN kernel size
		stride (int > 0): CNN kernel's stride
	"""
	def __init__(self, other_space={}):
		default_space = {
			'input_windows': hp.choice('input_windows', [5]),
			'topology': hp.choice('topology', [[5, 3]]),
			'kernel_size': hp.choice('kernel_size', [4]),
			'stride': hp.choice('stride', [1]),
			'dilation': hp.choice('dilation', [False]),
			'residual': hp.choice('residual', [False])
		}
		super(BinaryCNN, self).__init__({**default_space, **other_space})

	def make_model(self, params, num_inputs):
		window_size = num_inputs
		eff_history = window_size * params['input_windows']  								# Effective history = window_size * input_windows
		real_topology = window_size * np.array(params['topology'])							# Scale topology by the window size
		real_topology = np.clip(real_topology, a_min=1, a_max=None).astype(int)				# Make sure that layer outputs are always greater than zero
		mdl = CNN_Classifier(num_input_channels=1, channels=real_topology.tolist(), num_outputs=1, kernel_size=params['kernel_size'],
								stride=params['stride'], dilation=params['dilation'], residual=params['residual'])
		return mdl
