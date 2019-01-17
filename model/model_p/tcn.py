"""
Kevin Patel

Adapted from code by locuslab
Source: https://github.com/locuslab/TCN
"""
import sys
import os
import logging

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from common_util import MODEL_DIR
from model.common import PYTORCH_MODELS_DIR, ERROR_CODE, TEST_RATIO, VAL_RATIO
from model.model_p.binary_classifier import BinaryClassifier
from model.model_p.temporal_mixin import TemporalMixin


class Chomp1d(nn.Module):
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
		"""
		TCN Block Class

		DC: Dilated Casual Convolution
		WN: Weight Normalization
		RU: ReLu
		DO: Dropout

		--| DC |-->| WN |-->| RU |-->| DO |-->| DC |-->| WN |-->| RU |-->| DO |--(+)-->
		  |-------------------------|1x1 Conv (optional)|-------------------------|
		"""
	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
		super(TemporalBlock, self).__init__()
		self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
		self.chomp1 = Chomp1d(padding)
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)

		self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
		self.chomp2 = Chomp1d(padding)
		self.relu2 = nn.ReLU()
		self.dropout2 = nn.Dropout(dropout)

		self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
								 self.conv2, self.chomp2, self.relu2, self.dropout2)
		self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
		self.relu = nn.ReLU()
		self.init_weights()

	def init_weights(self):
		self.conv1.weight.data.normal_(0, 0.01)
		self.conv2.weight.data.normal_(0, 0.01)
		if self.downsample is not None:
			self.downsample.weight.data.normal_(0, 0.01)

	def forward(self, x):
		out = self.net(x)
		res = x if self.downsample is None else self.downsample(x)
		return self.relu(out + res)


class TemporalConvNet(nn.Module):
	def __init__(self, num_inputs, channels, kernel_size=2, stride=1, dropout=0.2):
		super(TemporalConvNet, self).__init__()
		layers = []
		num_levels = len(channels)
		for i in range(num_levels):
			dilation_size = 2 ** i
			in_channels = num_inputs if (i == 0) else channels[i-1]
			out_channels = channels[i]
			layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation_size,
									 padding=(kernel_size-1) * dilation_size, dropout=dropout)]

		self.network = nn.Sequential(*layers)

	def forward(self, x):
		return self.network(x)


class TCN(TemporalMixin, BinaryClassifier):
	"""
	Top level Temporal CNN Classifer.

	Parameters:
		history_multiplier (int > 0): Scalar to multiply input size by to get actual network input size (aka effective history)
		topology (list): Topology of the TCN divided by the actual network input size
		kernel_size (int > 1): CNN kernel size
		stride (int > 0): CNN kernel's stride 
		dropout (float [0, 1]): amount of dropout 
	"""
	def __init__(self, other_space={}):
		default_space = {
			'history_multiplier': hp.choice('history_multiplier', [3, 5, 10, 20]),
			'topology': hp.choice('topology', [[3], [3, 3], [3, 5, 1], [3, 1, 3], [5, 3, 1, .5]]),
			'kernel_size': hp.choice('kernel_size', [2, 4, 8]),
			'stride': hp.choice('stride', [1, 2]),
			'dropout': hp.uniform('dropout', .2, .8),
		}
		super(TCN, self).__init__({**default_space, **other_space})

	def make_model(self, params, num_inputs):
		real_num_inputs = num_inputs * params['history_multiplier']  								# Multiply by history_multiplier to get real expected inputs
		real_topology = real_num_inputs * np.array(params['topology'])								# Scale topology by the number of network inputs
		real_topology = np.clip(real_topology, a_min=1, a_max=None).astype(int)						# Make sure that layer outputs are always greater than zero
		mdl = TemporalConvNet(real_num_inputs, real_topology.tolist(), params['kernel_size'], params['stride'], params['dropout'])

		return mdl
