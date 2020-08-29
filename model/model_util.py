"""
Generic model utils
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
from kymatio.torch import Scattering1D as pyt_wavelet_scatter_1d

from common_util import is_type, assert_has_all_attr, is_valid, isnt, pairwise
from model.common import PYTORCH_ACT_MAPPING


# ********** HELPER FUNCTIONS **********
def assert_has_shape_attr(mod):
	assert is_type(mod, nn.Module), 'object must be a nn.Module'
	assert_has_all_attr(mod, 'in_shape', 'out_shape')

def log_prob_sigma(value, loc, log_scale):
	"""
	A slightly more stable (not confirmed yet) log prob taking in log_var instead of scale.
	modified from https://github.com/pytorch/pytorch/blob/2431eac7c011afe42d4c22b8b3f46dedae65e7c0/torch/distributions/normal.py#L65
	"""
	var = torch.exp(log_scale * 2)
	return (
		-((value - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
	)

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

def pyt_multihead_attention(W, q, k, v):
	"""
	Applies Pytorch Multiheaded Attention using existing MultiheadAttention object,
	assumes q, k, v tensors are shaped (batch, channels, sequence)
	Modified from: KurochkinAlexey/Recurrent-neural-processes

	Args:
		W (torch.nn.modules.activation.MultiheadAttention): pytorch Multihead attention object
		q (torch.tensor): query tensor, shaped like (Batch, Channels, Sequence)
		k (torch.tensor): key tensor, shaped like (Batch, Channels, Sequence)
		v (torch.tensor): value tensor, shaped like (Batch, Channels, Sequence)

	Returns:
		torch.tensor shaped like (Batch, Channels, Sequence)
	"""
	q = q.permute(2, 0, 1)
	k = k.permute(2, 0, 1)
	v = v.permute(2, 0, 1)
	o = W(q, k, v)[0]
	return o.permute(1, 2, 0).contiguous()


# ********** HELPER MODULES **********
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
		super().__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous()

class WaveletScatter1d(nn.Module):
	"""
	Apply One Dimensional Wavelet Scattering Transform on a vector and return the coefficients as a tuple of torch tensors.
	Wrapper around the Kymatio library, uses analytic (complex-valued) Morlet Wavelets down to 2nd order wavelets.
	arXiv:1812.11214
	"""
	def __init__(self, input_shape, max_scale_power, fo_wavelets):
		"""
		Args:
			input_shape (tuple): Length of the input vector
			max_scale_power (int): Maximum base 2 log scale of scattering transform
			fo_wavelets (int>=1): Number of first order wavelets per octave (2nd order is fixed to 1)
		"""
		super().__init__()
		self.scatter = pyt_wavelet_scatter_1d(max_scale_power, input_shape, fo_wavelets)
		#self.scatter.cuda()
		meta = self.scatter.meta()
		self.orders = [np.where(meta['order'] == i) for i in range(3)]

	def forward(self, x):
		Sx = self.scatter(x)
		coefficients = tuple(Sx[order] for order in self.orders)
		return coefficients


# ********** BLOCK MODULES **********
class OutputLinear(nn.Module):
	"""
	Adds a linear output layer or block to the end of an arbitrary embedding network.
	Used for Classification or Regression depending on the loss function used.
	"""
	def __init__(self, emb, out_shapes=[2], init_method='xavier_uniform'):
		"""
		Args:
			emb (nn.Module): embedding network to add output layer to
			out_shapes (list): output shape of the linear layer, if this has multiple numbers it this module will have multiple layers
		"""
		super().__init__()
		assert_has_shape_attr(emb)
		self.emb = emb
		out_net = [init_layer(nn.Linear(self.emb.out_shape[0], out_shapes[0]),
			act='linear', init_method=init_method)]

		if (len(out_shapes)>1):
			for lay_in, lay_out in pairwise(out_shapes):
				out_net.append(init_layer(nn.Linear(lay_in, lay_out), act='linear', init_method=init_method))

		self.out = nn.Sequential(*out_net)

	def forward(self, x):
		out_embedding = self.emb(x)
		if (len(self.emb.out_shape) == 1):
			out_linear = self.out(out_embedding)
		else:
			out_linear = self.out(out_embedding[:, :, -1])
		return out_linear

class TemporalLayer1d(nn.Module):
	"""
	Temporal convolutional layer (1d)

	DC: Dilated Casual Convolution
	WN: Weight Normalization
	RU: ReLu
	DO: Dropout
	"""
	def __init__(self, in_shape, out_shape, act, kernel_size, padding_size,
		dilation, dropout, init_method='xavier_uniform', chomp=False):
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
		super().__init__()
		self.in_shape, self.out_shape = in_shape, out_shape
		modules = nn.ModuleList()
		modules.append(
			init_layer(
				weight_norm(nn.Conv1d(self.in_shape[0], self.out_shape[0], kernel_size,
					stride=1, padding=padding_size, dilation=dilation, groups=1, bias=True)), # groups=1, groups=n, or groups=self.in_shape[0]
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
	def __init__(self, net, act, downsample_type='linear', init_method='xavier_uniform'):
		"""
		Args:
			net (nn.Module): nn.Module to wrap
			act (str): activation function
			init_method (str): layer weight initialization method
		"""
		super().__init__()
		assert_has_shape_attr(net)
		self.net = net
		self.out_act = PYTORCH_ACT_MAPPING.get(act)()
		self.downsample = None
		if (net.in_shape != net.out_shape):
			if (downsample_type == 'linear'):
				self.downsample = init_layer(nn.Linear(net.in_shape[0], net.out_shape[0], bias=True),
					act=act, init_method=init_method)
			elif (downsample_type == 'conv1d'):
				padding_size = max(int((net.out_shape[1] - net.in_shape[1])//2), 0)
				self.downsample = init_layer(nn.Conv1d(net.in_shape[0], net.out_shape[0],
					1, padding=padding_size, bias=True), act=act, init_method=init_method)

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


# ********** MODEL MODULES **********
class TemporalConvNet(nn.Module):
	"""
	Temporal ConvNet
	Builds logits of a TCN convolutional network

	Note: Receptive Field Size = Number TCN Blocks * Kernel Size * Last Layer Dilation Size
	"""
	def __init__(self, in_shape, pad_type='same', num_blocks=1,
		block_channels=[[5, 3, 5]], block_act='elu', out_act='relu',
		block_init='xavier_uniform', out_init='xavier_uniform', kernel_sizes=[3],
		dilation_index='global', global_dropout=.2, no_dropout=[0]):
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
		super().__init__()
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
				dropout = 0 if (is_valid(no_dropout) and i in no_dropout) else global_dropout
				out_width = TemporalLayer1d.get_out_width(layer_in_shape[1],
					kernel_size=kernel_size, dilation=dilation, padding_size=padding_size)
				layer_out_shape = (out_channels, out_width)
				layer = TemporalLayer1d(in_shape=layer_in_shape, out_shape=layer_out_shape,
					act=block_act, kernel_size=kernel_size, padding_size=padding_size,
					dilation=dilation, dropout=dropout, init_method=block_init)
				layers.append(('tl_{b}_{l}'.format(b=b, l=l, i=i), layer))
				layer_in_shape = layer_out_shape
				i += 1
			block_out_shape = layer_out_shape
			net = nn.Sequential(OrderedDict(layers))
			net.in_shape, net.out_shape = block_in_shape, block_out_shape
			#net.in_shape, net.out_shape = layers[0][1].in_shape, layers[-1][1].out_shape
			blocks.append(('rb_{b}'.format(b=b), ResidualBlock(net, out_act,
				downsample_type='conv1d', init_method=out_init)))
			block_in_shape = block_out_shape
		self.out_shape = block_out_shape
		self.convnet = nn.Sequential(OrderedDict(blocks))

	def forward(self, x):
		return self.convnet(x)

class FFN(nn.Module):
	"""
	MLP Feedforward Network
	"""
	def __init__(self, in_shape, out_shapes=[128, 128, 128], act='relu',
		init_method='xavier_uniform'):
		super().__init__()
		ffn_layers = []
		in_layer_shape = np.product(in_shape)
		for i, out_layer_shape in enumerate(out_shapes):
			ffn_layers.append(('ff_{i}'.format(i=i), init_layer(nn.Linear(in_layer_shape, out_layer_shape),
				act=act, init_method=init_method)))
			ffn_layers.append(('af_{i}'.format(i=i), PYTORCH_ACT_MAPPING.get(act)()))
			in_layer_shape = out_layer_shape

		self.ffn = nn.Sequential(OrderedDict(ffn_layers))

	def forward(self, x):
		return self.ffn(x)

class MultichannelFFN(nn.Module):
	"""
	Multi channel MLP Feedforward Network
	Cretes a FFN over independent channels (analagous to torch.nn.conv1d)
	"""
	def __init__(self, in_shape, out_shapes=[128, 128, 128], act='relu',
		init_method='xavier_uniform'):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor, expects a shape (in_channels, in_width)
		"""
		super().__init__()
		self.channel_ffn = [FFN(in_shape[1], out_shapes=out_shapes) for i in range(in_shape[0])]

	def forward(self, x):
		# TODO - iterate across channels and apply channel_ffn[i] to each and concatenate output
		return self.channel_ffn(x)

class AE(nn.Module):
	"""
	MLP Autoencoder
	"""
	def __init__(self, in_shape, out_shapes=[128, 64, 32, 16], act='relu',
		init_method='xavier_uniform'):
		super().__init__()
		encoder_layers = []
		in_ae = np.product(in_shape)
		in_lay_shape = in_ae
		for i, out_lay_shape in enumerate(out_shapes):
			encoder_layers.append(('ff_{i}'.format(i=i), init_layer(nn.Linear(in_lay_shape, out_lay_shape),
				act='linear', init_method=init_method)))
			encoder_layers.append(('af_{i}'.format(i=i), PYTORCH_ACT_MAPPING.get(act)()))
			in_lay_shape = out_lay_shape

		decoder_layers = []
		in_lay_shape = in_shape
		for i, (in_lay_shape, out_lay_shape) in enumerate(pairwise(reversed(out_shapes))):
			decoder_layers.append(('ff_{i}'.format(i=i), init_layer(nn.Linear(in_lay_shape, out_lay_shape),
				act='linear', init_method=init_method)))
			decoder_layers.append(('af_{i}'.format(i=i), PYTORCH_ACT_MAPPING.get(act)()))
		decoder_layers.append(('ff_{i}'.format(i=len(out_shapes)-1),
			init_layer(nn.Linear(out_shapes[0], in_ae), act='linear', init_method=init_method)))
		decoder_layers.append(('af_{i}'.format(i=len(out_shapes)-1), PYTORCH_ACT_MAPPING.get(act)()))

		self.encoder = nn.Sequential(OrderedDict(encoder_layers))
		self.decoder = nn.Sequential(OrderedDict(decoder_layers))

	def forward(self, x):
		return self.decoder(self.encoder(x))

