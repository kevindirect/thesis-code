"""
Kevin Patel
"""
import sys
import os
import logging

from common_util import MODEL_DIR
from model.common import PYTORCH_MODELS_DIR, ERROR_CODE, TEST_RATIO, VAL_RATIO
from model.model_p.binary_classifier import BinaryClassifier
from model.model_p.temporal_mixin import TemporalMixin
from model.model_p.tcn_util import TCN_Classifier


class BinaryTCN(TemporalMixin, BinaryClassifier):
	"""
	Top level Temporal CNN Binary Classifer.
	Note: Receptive Field Size = Number TCN Blocks * Kernel Size * Last Layer Dilation Size

	Parameters:
		num_windows (int > 0): Scalar to multiply input size by to get actual network input size (aka effective history)
		topology (list): Topology of the TCN divided by the actual network input size
		kernel_size (int > 1): CNN kernel size
		stride (int > 0): CNN kernel's stride 
		dropout (float [0, 1]): amount of dropout 
		attention (bool): whether or not to include attention block after each tcn block
		max_attn_len (int > 0): max length of attention (only relevant if attention is set to True)
	"""
	def __init__(self, other_space={}):
		default_space = {
			'num_windows': hp.choice('num_windows', [3, 5, 10, 20]),
			'topology': hp.choice('topology', [[3], [3, 3], [3, 5, 1], [3, 1, 3], [5, 3, 1, .5]]),
			'kernel_size': hp.choice('kernel_size', [2, 4, 8]),
			'stride': hp.choice('stride', [1, 2]),
			'dropout': hp.uniform('dropout', .2, .8),
			'attention': hp.choice('attention', [False]),
			'max_attn_len': hp.uniform('max_attn_len', 24, 120)
		}
		super(TCN, self).__init__({**default_space, **other_space})

	def make_model(self, params, num_inputs):
		real_num_inputs = num_inputs * params['num_windows']  								# Multiply by num_windows to get real expected inputs
		real_topology = real_num_inputs * np.array(params['topology'])						# Scale topology by the number of network inputs
		real_topology = np.clip(real_topology, a_min=1, a_max=None).astype(int)				# Make sure that layer outputs are always greater than zero
		mdl = TCN_Classifier(num_inputs=real_num_inputs, num_outputs=1, channels=real_topology.tolist(), kernel_size=params['kernel_size'], stride=params['stride'],
								dropout=params['dropout'], attention=params['attention'], max_attn_len=params['max_attn_len'])
		return mdl
