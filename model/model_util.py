"""
Kevin Patel
"""
import sys
import os
import logging
from math import floor
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from common_util import is_type, assert_has_all_attr, is_valid, isnt
from model.common import PYTORCH_ACT_MAPPING


# ********** HELPER FUNCTIONS **********
def assert_has_shape_attr(mod):
	assert is_type(mod, nn.Module), 'object must be a nn.Module'
	assert_has_all_attr(mod, 'in_shape', 'out_shape')

def init_layer(layer, act='linear', init_method='xavier_uniform'):
	"""
	Initialize layer weights
	"""
	def get_gain(act):
		"""
		Return recommended gain for activation function.
		"""
		try:
			gain = nn.init.calculate_gain(act)
		except ValueError as ve:
			if (act.endswith('elu')):
				gain = nn.init.calculate_gain('relu')
			else:
				gain = np.sqrt(2)
		return gain

	if (is_valid(layer)):
		if (init_method in ('zeros',)):
			nn.init.zeros_(layer.weight)
		elif (init_method in ('ones',)):
			nn.init.ones_(layer.weight)
		elif (init_method in ('eye', 'identity')):
			nn.init.eye_(layer.weight)
		elif (init_method in ('dirac')):
			nn.init.dirac_(layer.weight)

		# RANDOM METHODS:
		elif (init_method in ('normal',)):
			nn.init.normal_(layer.weight, mean=0, std=.01)
		elif (init_method in ('orthogonal',)):
			nn.init.orthogonal_(layer.weight, gain=get_gain(act))
		elif (init_method in ('xavier_uniform', 'glorot_uniform')):
			nn.init.xavier_uniform_(layer.weight, gain=get_gain(act))
		elif (init_method in ('xavier_normal', 'glorot_normal')):
			nn.init.xavier_normal_(layer.weight, gain=get_gain(act))
		elif (init_method in ('kaiming_uniform',)):
			act = 'leaky_relu' if (act == 'lrelu') else act
			try:
				nn.init.kaiming_uniform_(layer.weight, nonlinearity=act)
			except ValueError as ve:
				nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu') # fallback
		elif (init_method in ('kaiming_normal',)):
			act = 'leaky_relu' if (act == 'lrelu') else act
			try:
				nn.init.kaiming_normal_(layer.weight, nonlinearity=act)
			except ValueError as ve:
				nn.init.kaiming_normal_(layer.weight, nonlinearity='relu') # fallback
	return layer

def get_padding(conv_type, in_width, kernel_size, dilation=1, stride=1):
	return {
		'same': int((dilation*(kernel_size-1))//2),
		'full': int(dilation*(kernel_size-1))
	}.get(conv_type, 0)


# ********** MODEL CLASSES **********
class Chomp1d(nn.Module):
	"""
	This module is meant to guarantee causal convolutions for sequence modelling if the data set hasn't
	already been preprocessed to ensure this. If your labels/targets are already temporally shifted you
	don't need this.

	Prior to this step (kernel_size-1)*dilation units of padding are added to the left end of the layer,
	this module chops off this padding off the right (temporally recent) end.

	Only guaranteed to work if the stride is 1.
	"""
	def __init__(self, chomp_size):
		"""
		Args:
			chomp_size (int): size of input to cut off the right side of the input
		"""
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous()

class TemporalLayer1d(nn.Module):
	"""
	Temporal convolutional layer (1d)

	DC: Dilated Casual Convolution
	WN: Weight Normalization
	RU: ReLu
	DO: Dropout
	"""
	def __init__(self, in_shape, out_shape, act, kernel_size, padding_size, dilation, dropout, init_method='xavier_uniform', chomp=False):
		"""
		Args:
			in_shape (tuple): (in_channels, in_width) of the Conv1d layer
			out_shape (tuple): (out_channels, out_width) of the Conv1d layer
			act (str): layer activation
			kernel_size (int): size of the kernel
			padding_size (int): input padding size
			dilation (int): layer dilation
			dropout (float): probability of an element to be zeroed or dropped,
				uses AlphaDroput if the layer has a selu activation
			chomp (bool): whether to add a chomp step to make the network causal,
				if the data already ensures this (for example the labels are shifted)
				than this is not necessary. If it is set to true, 'padding_size' is removed off the
				right (temporally recent) end of the data.
			init_method (str): layer weight initialization method
		"""
		super(TemporalLayer1d, self).__init__()
		self.in_shape, self.out_shape = in_shape, out_shape
		modules = nn.ModuleList()
		modules.append(
			init_layer(
				weight_norm(nn.Conv1d(self.in_shape[0], self.out_shape[0], kernel_size, stride=1, padding=padding_size, dilation=dilation, groups=1, bias=True)), # groups=1, groups=n, or groups=self.in_shape[0]
				act=act,
				init_method=init_method
			)
		)
		if (chomp):
			modules.append(Chomp1d(padding_size))
		modules.append(PYTORCH_ACT_MAPPING.get(act)())
		modules.append(nn.AlphaDropout(dropout) if (act in ('selu',)) else nn.Dropout(dropout))
		self.layer = nn.Sequential(*modules)

	@classmethod
	def get_out_width(cls, in_width, kernel_size, dilation=1, padding_size=0, stride=1):
		return int(floor(((in_width+2*padding_size-dilation*(kernel_size-1)-1)/stride)+1))

	def forward(self, x):
		return self.layer(x)

class ResidualBlock(nn.Module):
	"""
	Wrap a residual connection around a network
	"""
	def __init__(self, net, act, init_method='xavier_uniform'):
		"""
		Args:
			net (nn.Module): nn.Module to wrap
			act (str): activation function
			init_method (str): layer weight initialization method
		"""
		super(ResidualBlock, self).__init__()
		assert_has_shape_attr(net)
		self.net = net
		padding_size = max(int((net.out_shape[1] - net.in_shape[1])//2), 0)
		self.downsample = init_layer(nn.Conv1d(net.in_shape[0], net.out_shape[0], 1, padding=padding_size, bias=True), act=act, init_method=init_method) if (net.in_shape != net.out_shape) else None
		self.out_act = PYTORCH_ACT_MAPPING.get(act)()

	def forward(self, x):
		residual = x if (isnt(self.downsample)) else self.downsample(x)
		try:
			net_out = self.net(x)
			return self.out_act(net_out + residual)
		except RuntimeError as e:
			print(e)
			logging.error('self.net(x).shape:  {}'.format(self.net(x).shape))
			logging.error('residual.shape: {}'.format(residual.shape))
			sys.exit(0)

class TemporalConvNet(nn.Module):
	"""
	Temporal ConvNet
	Builds logits of a TCN convolutional network

	Note: Receptive Field Size = Number TCN Blocks * Kernel Size * Last Layer Dilation Size
	"""
	def __init__(self, in_shape, pad_type='same', num_blocks=1, block_channels=[[5, 3, 5]], block_act='elu', out_act='relu', block_init='xavier_uniform', out_init='xavier_uniform', kernel_sizes=[3], dilation_index='global', global_dropout=.2, no_dropout=[0]):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor, expects a shape (in_channels, in_width)
			pad_type (str): padding method to use ('same' or 'full')
			num_blocks (int): number of residual blocks, each block consists of a tcn network and residual connection
			block_channels (list * list): cnn channel sizes in each block, or individual channel sizes per block in sequence
			block_act (str): activation function of each layer in each block
			out_act (str): output activation of each block
			block_init (str): layer weight initialization method of each layer in each block
			out_init (str): layer weight initialization method of each block
			kernel_sizes (list): list of CNN kernel sizes, must be the same length as block_channels
			dilation_index ('global'|'block'): what index to make each layer dilation a function of
			global_dropout (float): dropout probability of an element to be zeroed for any layer not in no_dropout
			no_dropout (list): list of global layer indices to disable dropout on
		"""
		super(TemporalConvNet, self).__init__()
		assert num_blocks >= len(block_channels), "list of block shapes have to be less than or equal to the number of blocks"
		assert num_blocks % len(block_channels) == 0, "number of block shapes must equally subdivide number of blocks"
		assert len(kernel_sizes) == len(block_channels), "number of kernels must be the same as the number of block channels"

		self.in_shape = block_in_shape = in_shape
		blocks = []
		i = 0
		for b in range(num_blocks):
			block_idx = b % len(block_channels)
			block_channel_list, kernel_size = block_channels[block_idx], kernel_sizes[block_idx] # XXX - param for residual connection from in_shape to all nodes?

			layer_in_shape = block_in_shape
			layers = []
			for l, out_channels in enumerate(block_channel_list):
				dilation = {
					'global': 2**i,
					'block': 2**l
				}.get(dilation_index)
				padding_size = get_padding(pad_type, layer_in_shape[1], kernel_size, dilation=dilation)
				dropout = 0 if (i in no_dropout) else global_dropout
				out_width = TemporalLayer1d.get_out_width(layer_in_shape[1], kernel_size=kernel_size, dilation=dilation, padding_size=padding_size)
				layer_out_shape = (out_channels, out_width)
				layer = TemporalLayer1d(in_shape=layer_in_shape, out_shape=layer_out_shape, act=block_act, kernel_size=kernel_size, padding_size=padding_size, dilation=dilation, dropout=dropout, init_method=block_init)
				layers.append(('TL_{b}_{l}'.format(b=b, l=l, i=i), layer))
				layer_in_shape = layer_out_shape
				i += 1
			block_out_shape = layer_out_shape
			net = nn.Sequential(OrderedDict(layers))
			net.in_shape, net.out_shape = block_in_shape, block_out_shape
			#net.in_shape, net.out_shape = layers[0][1].in_shape, layers[-1][1].out_shape
			blocks.append(('RB_{b}'.format(b=b), ResidualBlock(net, out_act, init_method=out_init)))
			block_in_shape = block_out_shape
		self.out_shape = block_out_shape
		self.convnet = nn.Sequential(OrderedDict(blocks))

	def forward(self, x):
		return self.convnet(x)


# ********** OUTPUT LAYER WRAPPER CLASSES **********
class OutputLinear(nn.Module):
	"""
	Adds a linear output layer to an arbitrary embedding network.
	Used for Classification or Regression depending on the loss function used.
	"""
	def __init__(self, emb, out_shape=2, init_method='xavier_uniform'):
		"""
		Args:
			emb (nn.Module): embedding network to add output layer to
			out_shape (tuple): output shape of the linear layer, should be C sized
		"""
		super(OutputLinear, self).__init__()
		assert_has_shape_attr(emb)
		self.emb = emb
		self.out = init_layer(nn.Linear(self.emb.out_shape[0], out_shape), act='linear', init_method=init_method)

	def forward(self, x):
		out_embedding = self.emb(x)
		if (len(self.emb.out_shape) == 1):
			out_linear = self.out(out_embedding)
		else:
			out_linear = self.out(out_embedding[:, :, -1])
		return out_linear

