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

from common_util import is_type, is_valid, isnt, odd_only, list_wrap, assert_has_all_attr, pairwise
from model.common import PYTORCH_ACT_MAPPING, PYTORCH_ACT1D_LIST, PYTORCH_INIT_LIST


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
class OutputBlock(nn.Module):
	"""
	Appends a feedforward layer or block to the end of an arbitrary embedding network.
	Used for Classification or Regression depending on the loss function used.
	"""
	def __init__(self, emb, out_shapes, act='linear', init='xavier_uniform'):
		"""
		Args:
			emb (nn.Module): embedding network to add output layer to
			out_shapes (int|list): output shape of the output block layer(s)
			act (str): activation function
			init (str): layer weight init method
		"""
		super().__init__()
		assert_has_shape_attr(emb)
		self.emb = emb
		self.out = FFN(self.emb.out_shape[0], out_shapes=list_wrap(out_shapes), \
			act=act, init=init)

	def forward(self, x):
		out_embedding = self.emb(x)
		if (len(self.emb.out_shape) == 1):
			output = self.out(out_embedding)
		else:
			output = self.out(out_embedding[:, :, -1])
		return output

	@classmethod
	def wrap(cls, emb):
		"""
		Wrap/append an OutputBlock to the passed embedding model
		if ob_out_shapes is not None and return original model otherwise
		"""
		if (hasattr(emb, 'ob_out_shapes') and is_valid(emb.ob_out_shapes)):
			model = OutputBlock(emb, out_shapes=emb.ob_out_shapes, act=emb.ob_act,
				init=emb.ob_init)
		else:
			model = emb
		return model

class TemporalLayer1d(nn.Module):
	"""
	Temporal convolutional layer (1d)

	DC: Dilated Casual Convolution
	WN: Weight Normalization
	RU: ReLu
	DO: Dropout
	"""
	def __init__(self, in_shape, out_shape, act, kernel_size, padding_size,
		dilation, dropout, init='xavier_uniform', chomp=False):
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
			init (str): layer weight initialization method
		"""
		super().__init__()
		self.in_shape, self.out_shape = in_shape, out_shape
		modules = nn.ModuleList()
		modules.append(
			init_layer(
				weight_norm(nn.Conv1d(self.in_shape[0], self.out_shape[0], kernel_size,
					stride=1, padding=padding_size, dilation=dilation, groups=1, bias=True)), # groups=1, groups=n, or groups=self.in_shape[0]
				act=act,
				init_method=init
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
	def __init__(self, net, act, downsample_type='linear', init='xavier_uniform'):
		"""
		Args:
			net (nn.Module): nn.Module to wrap
			act (str): activation function
			init (str): layer weight initialization method
		"""
		super().__init__()
		assert_has_shape_attr(net)
		self.net = net
		self.out_act = PYTORCH_ACT_MAPPING.get(act)()
		self.downsample = None
		if (net.in_shape != net.out_shape):
			if (downsample_type == 'linear'):
				self.downsample = init_layer(nn.Linear(net.in_shape[0], net.out_shape[0], bias=True),
					act=act, init_method=init)
			elif (downsample_type == 'conv1d'):
				padding_size = max(int((net.out_shape[1] - net.in_shape[1])//2), 0)
				self.downsample = init_layer(nn.Conv1d(net.in_shape[0], net.out_shape[0],
					1, padding=padding_size, bias=True), act=act, init_method=init)

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
	def __init__(self, in_shape, block_channels=[[5, 5, 5]], num_blocks=1,
		kernel_sizes=3, dropouts=0.0, global_dropout=.5, global_dilation=True,
		block_act='elu', out_act='relu', block_init='xavier_uniform',
		out_init='xavier_uniform', pad_type='same',
		ob_out_shapes=None, ob_act='linear', ob_init='xavier_uniform'):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor,
				expects a shape (in_channels, in_width)
			block_channels (list * list): cnn channel sizes in each block,
				or individual channel sizes per block in sequence
			num_blocks (int): number of residual blocks,
				each block consists of a tcn network and residual connection
			kernel_sizes (int|list): list of CNN kernel sizes,
				if a list its length must be either 1 or len(block_channels)
			dropouts (float|list): dropout probability ordered by global index
				if a single floating point value, sets dropout for first layer only
				if None, uses global_dropout everywhere
				if the list size doesn't match the network layer count,
					uses global_dropout for all layers past the list size.
				if a particular element is None, uses global_dropout at that index
			global_dropout (float): default dropout probability
			global_dilation (bool): whether to use global or block indexed dilation
			block_act (str): activation function of each layer in each block
			out_act (str): output activation of each block
			block_init (str): hidden layer weight initialization method
			out_init (str): output layer weight initialization method
			pad_type ('same'|'full'): padding method to use
			ob_out_shapes
			ob_act
			ob_init
			"""
		super().__init__()
		self.in_shape = block_in_shape = in_shape
		kernel_sizes, dropouts = list_wrap(kernel_sizes), list_wrap(dropouts)
		blocks = []
		i = 0

		assert num_blocks >= len(block_channels), \
			'list of block shapes have to be less than or equal to the number of blocks'
		assert num_blocks % len(block_channels) == 0, \
			'number of block shapes must equally subdivide number of blocks'
		assert len(kernel_sizes) == len(block_channels), \
			'number of kernels must be the same as the number of block channels'

		for b in range(num_blocks):
			# XXX - param for residual connection from in_shape to all nodes?
			block_idx = b % len(block_channels)
			block_channel_list = block_channels[block_idx]
			kernel_size = kernel_sizes[block_idx]
			layer_in_shape = block_in_shape
			layers = []

			for l, out_channels in enumerate(block_channel_list):
				dilation = {
					True: 2**i,
					False: 2**l
				}.get(global_dilation)
				padding_size = get_padding(pad_type, layer_in_shape[1], kernel_size,
					dilation=dilation)
				out_width = TemporalLayer1d.get_out_width(layer_in_shape[1],
					kernel_size=kernel_size, dilation=dilation, padding_size=padding_size)
				layer_out_shape = (out_channels, out_width)
				dropout = global_dropout \
					if (isnt(dropouts) or i >= len(dropouts) or isnt(dropouts[i])) \
					else dropouts[i]
				layer = TemporalLayer1d(in_shape=layer_in_shape, out_shape=layer_out_shape,
					act=block_act, kernel_size=kernel_size, padding_size=padding_size,
					dilation=dilation, dropout=dropout, init=block_init)
				layers.append(('tl_{b}_{l}'.format(b=b, l=l, i=i), layer))
				layer_in_shape = layer_out_shape
				i += 1
			block_out_shape = layer_out_shape
			net = nn.Sequential(OrderedDict(layers))
			net.in_shape, net.out_shape = block_in_shape, block_out_shape
			#net.in_shape, net.out_shape = layers[0][1].in_shape, layers[-1][1].out_shape
			blocks.append(('rb_{b}'.format(b=b), ResidualBlock(net, out_act,
				downsample_type='conv1d', init=out_init)))
			block_in_shape = block_out_shape
		self.out_shape = block_out_shape
		self.model = nn.Sequential(OrderedDict(blocks))

		# GenericModel uses these params to to append an OutputBlock to the model
		self.ob_out_shapes = ob_out_shapes	# If this is None, no OutputBlock is added
		self.ob_act, self.ob_init = ob_act, ob_init


	def forward(self, x):
		return self.model(x)

class StackedTCN(TemporalConvNet):
	"""
	Wrapper Module for model_util.TemporalConvNet,
	creates a fixed width, single block TCN.
	"""
	def __init__(self, in_shape, size=128, depth=3, kernel_sizes=3,
		input_dropout=0.0, output_dropout=0.0, global_dropout=.5,
		global_dilation=True, block_act='elu', out_act='relu',
		block_init='xavier_uniform', out_init='xavier_uniform', pad_type='full',
		ob_out_shapes=None, ob_act='linear', ob_init='xavier_uniform'):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor (C, S)
			size (int): network embedding size
			depth (int): number of hidden layers to stack
			kernel_sizes (int|list): list of CNN kernel sizes,
				if a list its length must be either 1 or depth
			input_dropout (float): first layer dropout
			output_dropout (float): last layer dropout
			global_dropout (float): default dropout probability
			global_dilation (bool): whether to use global or block indexed dilation
			block_act (str): activation function of each layer in each block
			out_act (str): output activation of each block
			block_init (str): hidden layer weight initialization method
			out_init (str): output layer weight initialization method
			pad_type ('same'|'full'): padding method to use
			ob_out_shapes
			ob_act
			ob_ini
		"""
		dropouts = [None] * depth
		dropouts[0], dropouts[-1] = input_dropout, output_dropout
		super().__init__(in_shape, block_channels=[[size] * depth], num_blocks=1,
			kernel_sizes=kernel_sizes, dropouts=dropouts,
			global_dropout=global_dropout, global_dilation=global_dilation,
			block_act=block_act, out_act=out_act, block_init=block_init,
			out_init=out_init, pad_type=pad_type,
			ob_out_shapes=ob_out_shapes, ob_act=ob_act, ob_init=ob_init)

	@classmethod
	def suggest_params(cls, trial=None, label_size=1, add_ob=False):
		"""
		suggest model hyperparameters from an optuna trial object
		or return fixed default hyperparameters
		"""
		if (is_valid(trial)):
			params = {
				'size': trial.suggest_int('size', 2**5, 2**8),
				'depth': trial.suggest_int('depth', 2, 6),
				'kernel_sizes': odd_only(trial.suggest_int('kernel_sizes', 3, 15)),
				'input_dropout': trial.suggest_uniform('input_dropout', 0.0, 1.0),
				'output_dropout': trial.suggest_uniform('output_dropout', 0.0, 1.0),
				'global_dropout': trial.suggest_uniform('global_dropout', 0.0, 1.0),
				'global_dilation': True,
				'block_act': trial.suggest_categorical('block_act', PYTORCH_ACT1D_LIST),
				'out_act': trial.suggest_categorical('out_act', PYTORCH_ACT1D_LIST),
				'block_init': trial.suggest_categorical('block_init', PYTORCH_INIT_LIST),
				'out_init': trial.suggest_categorical('out_init', PYTORCH_INIT_LIST),
				'pad_type': trial.suggest_categorical('pad_type', ('same', 'full')),
				'label_size': label_size,
				'ob_out_shapes': label_size+1 if (add_ob) else None,
				'ob_act': trial.suggest_categorical('ob_act', PYTORCH_ACT1D_LIST),
				'ob_init': trial.suggest_categorical('ob_init', PYTORCH_INIT_LIST)
			}
		else:
			params = {
				'size': 128,
				'depth': 3,
				'kernel_sizes': 3,
				'input_dropout': 0.0,
				'output_dropout': 0.0,
				'global_dropout': .5,
				'global_dilation': True,
				'block_act': 'elu',
				'out_act': 'relu',
				'block_init': 'xavier_uniform',
				'out_init': 'xavier_uniform',
				'pad_type': 'full',
				'label_size': label_size,
				'ob_out_shapes': label_size+1 if (add_ob) else None,
				'ob_act': 'linear',
				'ob_init': 'xavier_uniform'
			}
		return params

class FFN(nn.Module):
	"""
	MLP Feedforward Network
	"""
	def __init__(self, in_shape, out_shapes=[128, 128, 128], act='relu',
		init='xavier_uniform'):
		super().__init__()
		io_shapes = [np.product(in_shape)]
		io_shapes.extend(out_shapes)
		ffn_layers = []
		for l, (i, o) in enumerate(pairwise(io_shapes)):
			ffn_layers.append((f'ff_{l}', init_layer(nn.Linear(i, o), act=act, \
				init_method=init)))
			if (act not in ('linear', None)):
				ffn_layers.append((f'af_{l}', PYTORCH_ACT_MAPPING.get(act)()))

		self.model = nn.Sequential(OrderedDict(ffn_layers))

	def forward(self, x):
		return self.model(x)

class MultichannelFFN(nn.Module):
	"""
	Multi channel MLP Feedforward Network
	Creates a FFN over independent channels (analagous to torch.nn.conv1d)
	"""
	def __init__(self, in_shape, out_shapes=[128, 128, 128], act='relu',
		init='xavier_uniform'):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor, expects a shape (in_channels, in_width)
		"""
		super().__init__()
		raise NotImplementedError()
		self.model = [FFN(in_shape[1], out_shapes=out_shapes) for i in range(in_shape[0])]

	def forward(self, x):
		# TODO - iterate across channels and apply model[i] to each and concatenate output
		return self.model(x)

class AE(nn.Module):
	"""
	MLP Autoencoder
	"""
	def __init__(self, in_shape, out_shapes=[128, 64, 32, 16], act='relu',
		init='xavier_uniform'):
		super().__init__()
		encoder_layers = []
		in_ae = np.product(in_shape)
		in_lay_shape = in_ae
		for i, out_lay_shape in enumerate(out_shapes):
			encoder_layers.append(('ff_{i}'.format(i=i), init_layer(nn.Linear(in_lay_shape, out_lay_shape),
				act='linear', init_method=init)))
			encoder_layers.append(('af_{i}'.format(i=i), PYTORCH_ACT_MAPPING.get(act)()))
			in_lay_shape = out_lay_shape

		decoder_layers = []
		in_lay_shape = in_shape
		for i, (in_lay_shape, out_lay_shape) in enumerate(pairwise(reversed(out_shapes))):
			decoder_layers.append(('ff_{i}'.format(i=i), init_layer(nn.Linear(in_lay_shape, out_lay_shape),
				act='linear', init_method=init)))
			decoder_layers.append(('af_{i}'.format(i=i), PYTORCH_ACT_MAPPING.get(act)()))
		decoder_layers.append(('ff_{i}'.format(i=len(out_shapes)-1),
			init_layer(nn.Linear(out_shapes[0], in_ae), act='linear', init_method=init)))
		decoder_layers.append(('af_{i}'.format(i=len(out_shapes)-1), PYTORCH_ACT_MAPPING.get(act)()))

		self.encoder = nn.Sequential(OrderedDict(encoder_layers))
		self.decoder = nn.Sequential(OrderedDict(decoder_layers))
		self.model = nn.Sequential(self.encoder, self.decoder)

	def forward(self, x):
		return self.model(x)

