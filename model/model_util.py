"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common_util import identity_fn
from model.common import EXPECTED_NUM_HOURS


class TemporalBlock(nn.Module):
	"""
	TCN Block Class

	DC: Dilated Casual Convolution
	WN: Weight Normalization
	RU: ReLu
	DO: Dropout

	----| DC |-->| WN |-->| RU |-->| DO |-->| DC |-->| WN |-->| RU |-->| DO |--(+)---->
		|-------------------------|1x1 Conv (optional)|-------------------------|

	May not work correctly for a stride greater than 1
	"""
	def __init__(self, act, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
		super(TemporalBlock, self).__init__()
		self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
		self.chomp1 = Chomp1d(padding)
		self.act1 = act()
		self.dropout1 = nn.Dropout(dropout)

		#self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
		#self.chomp2 = Chomp1d(padding)
		#self.relu2 = nn.ReLU()
		#self.dropout2 = nn.Dropout(dropout)

		#self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
		self.net = nn.Sequential(self.conv1, self.chomp1, self.act1, self.dropout1)
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
			#self.conv2.weight.data.normal_(0, 0.01)
			if (self.downsample is not None):
				self.downsample.weight.data.normal_(0, 0.01)

		elif (init_method in ('xavier_uniform', 'glorot_uniform')):
			nn.init.xavier_uniform_(self.conv1.weight, gain=np.sqrt(2))
			#nn.init.xavier_uniform_(self.conv2.weight, gain=np.sqrt(2))
			if (self.downsample is not None):
				nn.init.xavier_uniform_(self.downsample.weight, gain=np.sqrt(2))

	def forward(self, x):
		out = self.net(x)
		res = x if (self.downsample is None) else self.downsample(x)
		try:
			comb = self.relu(out + res)
		except RuntimeError as e:
			print(e)
			print('out.shape: {}'.format(out.shape))
			print('res.shape: {}'.format(res.shape))
			sys.exit(0)
		return comb


class Chomp1d(nn.Module):
	"""
	This transform is meant to chop off any trailing elements caused by 1d convolution padding, only guaranteed to work if the stride is 1.
	"""
	def __init__(self, chomp_size, chomp_loc=0):
		"""

		Args:
			chomp_size: size of input to cut off
			chomp_loc: start location of cut
		"""
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous() # XXX - chomps off (kernel_size-1)*dilation off right end


class TemporalLayer(nn.Module):
	"""
	DC: Dilated Casual Convolution
	WN: Weight Normalization
	RU: ReLu
	DO: Dropout
	"""
	def __init__(self, act, n_inputs, n_outputs, kernel_size, dilation, padding, stride, dropout):
		super(TemporalLayer, self).__init__()
		self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
		self.chomp = Chomp1d(padding)
		self.act = act()
		self.dropout = nn.Dropout(dropout)

		self.layer = nn.Sequential(self.conv, self.chomp, self.act, self.dropout)
		self.init_weights()

	def init_weights(self, init_method='xavier_uniform'):
		"""
		Initialize convolutional layer weights.
		"""
		if (init_method == 'normal'):
			self.conv.weight.data.normal_(0, 0.01)
		elif (init_method in ('xavier_uniform', 'glorot_uniform')):
			nn.init.xavier_uniform_(self.conv.weight, gain=np.sqrt(2))

	def forward(self, x):
		return self.layer(x)


class TemporalBlock(nn.Module):
	def __init__(self):
		super(TemporalBlock, self).__init__()



class TemporalConvNet(nn.Module):
	"""
	Temporal ConvNet
	Builds logits of a TCN convolutional network
	"""
	def __init__(self, input_shape, num_blocks=1, block_shapes=[[5, 3, 5]], kernel_size=[3], block_act='relu', out_act='relu', dropout=.8):
		"""

		Args:
			input_shape (tuple):
			num_blocks (int): number of network blocks,
				each block consists of a tcn network and residual connection
			block_shapes (list): topology of the tcn network of each block,
				or of individual blocks if they differ
			kernel_sizes (list): kernel sizes, must be the same length as block_shapes
			block_act (str): activation function of each temporal convolution
			out_act (str): block output activation function
			dropout (float): global dropout parameter
		"""
		super(TemporalConvNet, self).__init__()
		assert num_blocks >= len(block_shapes), "list of block shapes have to be less than or equal to the number of blocks"
		assert num_blocks % len(block_shapes) == 0, "number of block shapes must equally subdivide number of blocks"
		assert len(kernel_sizes) == len(block_shapes), "number of kernels must be the same as the number of block shapes"
		# assert block_shapes[0]

		block_input = input_shape
		for b in range(num_blocks):
			block_idx = b % len(block_shapes)
			block_shape, kernel_size = block_shapes[block_idx], kernel_sizes[block_idx]
			# XXX - Parameter for residual connection from input to all nodes?
			# XXX - disable dropout in first layer?

			layer_input = block_input
			block_output = block_shape[-1]

			for i, layer_output in enumerate(block_shape):
				dilation_size = 2**i
				padding_size = (kernel_size-1) * dilation_size # XXX - what is this?, shouldn't it depend on the block input size?
				# set input channels
				# set output channels based on block_shape[i]
				TemporalLayer(layer_input, layer_output, kernel_size, dilation_size, padding_size, 1, dropout)
				layer_input = layer_output
			block_input = block_output


	def forward(self, x):
		return self.network(x)


