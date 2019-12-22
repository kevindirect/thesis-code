"""
Kevin Patel
"""
import sys
import os
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from common_util import is_type, assert_has_all_attr, is_valid, isnt
from model.common import PYTORCH_ACT_MAPPING


# ********** FUNCTIONS **********
def assert_has_shape_attr(mod):
	assert is_type(mod, nn.Module), "object must be a nn.Module"
	assert_has_all_attr(mod, "in_shape", "out_shape")

def init_layer(layer, act='relu', init_method='xavier_uniform'):
	"""
	Initialize layer weights
	"""
	if (is_valid(layer)):
		if (init_method == 'normal'):
			#layer.weight.data.normal_(0, 0.01)
			nn.init.normal_(layer.weight, mean=0, std=.01)
		elif (init_method in ('xavier_uniform', 'glorot_uniform')):
			try:
				gain = nn.init.calculate_gain(act)
			except ValueError as ve:
				gain = np.sqrt(2)
			nn.init.xavier_uniform_(layer.weight, gain=gain)
	return layer


# ********** CLASSES **********
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
	def __init__(self, in_shape, out_shape, act, kernel_size, padding_size, dilation, dropout, chomp=False):
		"""
		Args:
			in_shape (int): input channels of the Conv1d layer
			out_shape (int): output channels of the Conv1d layer
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
		"""
		super(TemporalLayer1d, self).__init__()
		self.in_shape, self.out_shape = in_shape, out_shape
		modules = nn.ModuleList()
		modules.append(
			init_layer(
				weight_norm(nn.Conv1d(self.in_shape, self.out_shape, kernel_size, stride=1, padding=padding_size, dilation=dilation, groups=1)) # XXX groups=1, groups=n, or groups=self.in_shape
			)
		)
		if (chomp):
			modules.append(Chomp1d(padding_size))
		modules.append(PYTORCH_ACT_MAPPING.get(act)())
		modules.append(nn.AlphaDropout(dropout) if (act in ('selu',)) else nn.Dropout(dropout))
		self.layer = nn.Sequential(*modules)

	def forward(self, x):
		return self.layer(x)

class ResidualBlock(nn.Module):
	"""
	Wrap a residual connection around a network
	"""
	def __init__(self, net, act):
		"""
		Args:
			net (nn.Module): nn.Module to wrap
			act (str): activation function
		"""
		super(ResidualBlock, self).__init__()
		assert_has_shape_attr(net)
		self.net = net
		self.downsample = init_layer(nn.Conv1d(net.in_shape, net.out_shape, 1)) if (net.in_shape != net.out_shape) else None
		self.out_act = PYTORCH_ACT_MAPPING.get(act)()

	def forward(self):
		residual = x if (isnt(self.downsample)) else self.downsample(x)
		try:
			return self.out_act(self.net(x) + residual)
		except RuntimeError as e:
			print(e)
			logging.error('self.net(x).shape:  {}'.format(self.net(x).shape))
			loggine.error('residual.shape: {}'.format(residual.shape))
			sys.exit(0)

class TemporalConvNet(nn.Module):
	"""
	Temporal ConvNet
	Builds logits of a TCN convolutional network

	Note: Receptive Field Size = Number TCN Blocks * Kernel Size * Last Layer Dilation Size
	"""
	def __init__(self, in_shape, num_blocks=1, block_shapes=[[5, 3, 5]], block_act='elu', out_act='relu', kernel_sizes=[3], dilation_index='global', global_dropout=.2, no_dropout=[0]):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor, expects a shape (Channels_in, Length_in)
			num_blocks (int): number of residual blocks, each block consists of a tcn network and residual connection
			block_shapes (list * list): shape of cnn layers in each block, or individual shape per block in sequence
			block_act (str): activation function of each layer in each block
			out_act (str): output activation of each block
			kernel_sizes (list): list of CNN kernel sizes, must be the same length as block_shapes
			dilation_index ('global'|'block'): what index to make each layer dilation a function of
			global_dropout (float): dropout probability of an element to be zeroed for any layer not in no_dropout
			no_dropout (list): list of global layer indices to disable dropout on
		"""
		super(TemporalConvNet, self).__init__()
		assert num_blocks >= len(block_shapes), "list of block shapes have to be less than or equal to the number of blocks"
		assert num_blocks % len(block_shapes) == 0, "number of block shapes must equally subdivide number of blocks"
		assert len(kernel_sizes) == len(block_shapes), "number of kernels must be the same as the number of block shapes"

		block_input = in_shape[0] # Get the input tensor's channel size
		blocks = []
		i = 0
		for b in range(num_blocks):
			block_idx = b % len(block_shapes)
			block_shape, kernel_size = block_shapes[block_idx], kernel_sizes[block_idx] # XXX - param for residual connection from input to all nodes?

			layer_input = block_input
			block_output = block_shape[-1]
			layers = []
			for l, layer_output in enumerate(block_shape):
				dilation = {
					'global': 2**i,
					'block': 2**l
				}.get(dilation_index)
				padding_size = 0 #(kernel_size-1) * dilation # XXX (Unneeded?) One "step" of the kernel will be needed for padding
				dropout = 0 if (i in no_dropout) else global_dropout
				name = 'TL_{b}_{l}'.format(b=b, l=l, i=i)
				layer = TemporalLayer1d(in_shape=layer_input, out_shape=layer_output, act=block_act, kernel_size=kernel_size, padding_size=padding_size, dilation=dilation, dropout=dropout)
				layers.append((name, layer))
				layer_input = layer_output
				i += 1
			net = nn.Sequential(OrderedDict(layers))
			net.in_shape, net.out_shape = layers[0][1].in_shape, layers[-1][1].out_shape
			blocks.append(ResidualBlock(net, out_act))
			block_input = block_output
		self.in_shape, self.out_shape = in_shape, block_output
		self.convnet = nn.Sequential(*blocks)

	def forward(self, x):
		return self.convnet(x)

class Classifier(nn.Module):
	"""
	Adds a logistic regression output layer to an arbitrary embedding network
	"""
	def __init__(self, emb, out_shape=1):
		"""
		Args:
			emb (nn.Module): embedding network to add output layer to
			out_shape (tuple): output shape of the linear layer
		"""
		super(Classifier, self).__init__()
		assert_has_shape_attr(emb)
		self.emb = emb
		self.out = nn.Linear(emb.out_shape, out_shape)
		self.logprob = nn.LogSoftmax(dim=1)

	def forward(self, x):
		out_embedding = self.emb(x)
		out_score = self.out(out_embedding[:, :, -1])
		out_prob = self.logprob(out_score)
		return out_prob

class Regressor(nn.Module):
	"""
	Adds a linear regression output layer to an arbitrary embedding network
	"""
	def __init__(self, emb, out_shape=1):
		"""
		Args:
			emb (nn.Module): embedding network to add output layer to
			out_shape (tuple): output shape of the linear layer
		"""
		super(Regressor, self).__init__()
		assert_has_shape_attr(emb)
		self.emb = emb
		self.out = nn.Linear(emb.out_shape, out_shape)

	def forward(self, x):
		out_embedding = self.emb(x)
		out_linear = self.out(out_embedding[:, :, -1])
		return out_linear

