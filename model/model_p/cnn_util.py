"""
Kevin Patel

Adapted from code by locuslab and flrngel
Source: https://github.com/locuslab/TCN, https://github.com/flrngel/TCN-with-attention
"""
import sys
import os
from math import floor
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from common_util import MODEL_DIR
from model.common import PYTORCH_MODELS_DIR, ERROR_CODE, TEST_RATIO, VAL_RATIO


class Chomp1d(nn.Module):
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous() # XXX - chomps off (kernel_size-1)*dilation off right end


class ConvBlock(nn.Module):
	"""
	Conv Block Class
	"""
	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, residual):
		super(ConvBlock, self).__init__()
		self.residual = residual
		self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
		# self.chomp1 = Chomp1d(padding)
		self.relu1 = nn.ReLU()
		self.net = nn.Sequential(self.conv1, self.relu1)

		if (self.residual):
			self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if (n_inputs != n_outputs) else None
			self.relu = nn.ReLU()
		self.init_weights()

	def init_weights(self, init_method='xavier_uniform'):
		"""
		Initialize convolutional layer weights.
		TCN networks are influenced heavily by weight initialization.
		"""
		if (init_method == 'normal'):
			self.conv1.weight.data.normal_(0, 0.01)
			if (self.residual and self.downsample is not None):
				self.downsample.weight.data.normal_(0, 0.01)

		elif (init_method in ('xavier_uniform', 'glorot_uniform')):
			nn.init.xavier_uniform_(self.conv1.weight, gain=np.sqrt(2))
			if (self.residual and self.downsample is not None):
				nn.init.xavier_uniform_(self.downsample.weight, gain=np.sqrt(2))

	def forward(self, x):
		out = self.net(x)
		if (self.residual):
			res = x if (self.downsample is None) else self.downsample(x)
			try:
				comb = self.relu(out + res)
			except RuntimeError as e:
				print(e)
				print('out.shape: {}'.format(out.shape))
				print('res.shape: {}'.format(res.shape))
				sys.exit(0)
			return comb
		else:
			return out


class ConvNet(nn.Module):
	"""
	Conv Net.
	This class can be used to build classifiers and regressors.
	"""
	def __init__(self, num_input_channels, channels, kernel_size, stride, dilation, residual):
		super(ConvNet, self).__init__()
		layers, num_levels = [], len(channels)

		for i in range(num_levels):
			in_channels = num_input_channels if (i == 0) else channels[i-1]
			out_channels = channels[i]
			dilation_size = 2 ** i if (dilation) else 1
			layers += [ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, residual=residual)]

		self.network = nn.Sequential(*layers)

	def forward(self, x):
		return self.network(x)


class CNN_Classifier(nn.Module):
	"""
	CNN Classifier Network

	Args:
		num_input_channels (int): number of input channels
		channels (list): list of output channel sizes in order from first to last
		num_outputs (int): number of outputs, usually the number of classes to predict (defaults to binary case)
		kernel_size (int > 1): CNN kernel size
		stride (int > 0): CNN kernel's stride
		dilation (bool): use dilation
		residual (bool): add residual connection
	"""
	def __init__(self, num_input_channels, channels, num_outputs=1, kernel_size=2, stride=1, dilation=False, residual=False):
		super(CNN_Classifier, self).__init__()
		self.cnn = ConvNet(num_input_channels, channels, kernel_size=kernel_size, stride=stride, dilation=dilation, residual=residual)
		pool_kernel_size = kernel_size
		pool_stride = 2
		pool_padding = 0
		pool_seqlen = int(floor(((channels[-1] + 2*pool_padding - pool_kernel_size) / pool_stride) + 1))
		self.pool = nn.AvgPool1d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding, ceil_mode=False)
		self.linear = nn.Linear(channels[-1], num_outputs)

	def forward(self, x):
		"""
		Input must have have shape (N, C_in, L_in) where
			N: number of batches
			C_in: number of input channels
			L_in: length of input sequence

		Output shape will be (N, C_out) where
			N: number of batches
			C_out: number of classes
		"""
		out_embedded = self.cnn(x)
		out_pooled = self.pool(out_embedded)
		out = self.linear(out_pooled[:, :, -1])
		return out
