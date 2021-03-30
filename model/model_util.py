"""
Generic model utils
Kevin Patel
"""
import sys
import os
import logging
from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from kymatio.torch import Scattering1D as pt_wavelet_scatter_1d

from common_util import is_type, is_valid, isnt, list_wrap, assert_has_all_attr, pairwise, odd_only
from model.common import PYTORCH_ACT_MAPPING, PYTORCH_ACT1D_LIST, PYTORCH_INIT_LIST


# ********** HELPER FUNCTIONS **********
def assert_has_shape_attr(mod):
	assert is_type(mod, nn.Module), 'object must be a torch.nn.Module'
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

def get_padding(pad_mode, in_width, kernel_size, dilation=1, stride=1):
	return {
		'same': dilation*(kernel_size-1),
		'full': dilation*(kernel_size-1)*2
	}.get(pad_mode, 0)

def pt_multihead_attention(W, q, k, v):
	"""
	Applies Pytorch Multiheaded Attention using existing MultiheadAttention object,
	assumes q, k, v tensors are shaped (batch, channels, sequence)
	Modified from: KurochkinAlexey/Recurrent-neural-processes
	
	Attention(Q, K, V) = softmax(QK'/sqrt(d_k)) V

	MH(Q, K, V) = Concat[head1; head2; ...; headn] Wo

		where head[i] = Attention(Q Wq[i], K Wk[i], V Wv[i])

	Args:
		W (torch.nn.modules.activation.MultiheadAttention): pytorch Multihead attention object
		q (torch.tensor): query tensor, shaped like (Batch, Channels, Sequence)
		k (torch.tensor): key tensor, shaped like (Batch, Channels, Sequence)
		v (torch.tensor): value tensor, shaped like (Batch, Channels, Sequence)

	Returns:
		torch.tensor shaped like (Batch, Channels, Sequence) and attention weights
	"""
	q = q.permute(2, 0, 1)
	k = k.permute(2, 0, 1)
	v = v.permute(2, 0, 1)
	o, w = W(q, k, v)	# MHA forward pass
	return o.permute(1, 2, 0).contiguous(), w


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
		self.scatter = pt_wavelet_scatter_1d(max_scale_power, input_shape, fo_wavelets)
		#self.scatter.cuda()
		meta = self.scatter.meta()
		self.orders = [np.where(meta['order'] == i) for i in range(3)]

	def forward(self, x):
		Sx = self.scatter(x)
		coefficients = tuple(Sx[order] for order in self.orders)
		return coefficients

class Apply(nn.Module):
	"""
	Module wrapper to apply a function to the input.
	"""
	def __init__(self, fn):
		super().__init__()
		self.fn = fn

	def forward(self, x):
		return self.fn(x)


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
		self.out = FFN(self.emb.out_shape, out_shapes=list_wrap(out_shapes), \
			act=act, init=init)

	def forward(self, x):
		out_embed = self.emb(x)
		out_score = self.out(out_embed)
		return out_score

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

class TemporalLayer2d(nn.Module):
	"""
	Temporal convolutional layer (2d)

	The network performs 2D convolutions but only over the temporal (width) dimension.
	All kernel size, dilation, and padding size parameters only affect the temporal
	dimension of the kernel or input.

	This module inputs/outputs a tensor shaped like (N, C, H, W),
	where:
		* N: batch
		* C: channel
		* H: height
		* W: width

	For a multivate time series input the height dimension indexes the series
	and the width dimension indexes time.

	To ensure convolution over the temporal (width) dimension only,
	the following height settings are fixed:
		* The kernel height is always the height of the input
		* The kernel dilation height is one (no dilation over height)
		* The input padding height is zero (no padding height)
               _____________    ____________    _________________    ____________    _____________
	x -----|__padding__|----|__conv2d__|----|__weight_norm__|----|__act_fn__|----|__dropout__|-----> temporal_layer_2d(x)

	This layer performs the following operations in order:
		1. 2D Dilated Causal Convolution
		2. Weight Normalization
		3. ReLu / other activation
		4. Dropout
	"""
	def __init__(self, in_shape, out_shape, act, kernel_size, padding_size,
		dilation, dropout, init='xavier_uniform'):
		"""
		Args:
			in_shape (tuple): (in_channels, in_height, in_width) of the Conv2d layer
			out_shape (tuple): (out_channels, in_height, out_width) of the Conv2d layer
			act (str): layer activation
			kernel_size (int): width of the kernel (height is the height of input)
			padding_size (int): input padding width (padding height is zero)
			dilation (int): layer dilation width (no dilation over height dimension)
			dropout (float): probability of an element to be zeroed or dropped,
				uses AlphaDroput if the layer has a selu activation
			init (str): layer weight initialization method
		"""
		super().__init__()
		self.in_shape, self.out_shape = in_shape, out_shape
		pad_l = padding_size//2
		pad_r = padding_size - pad_l
		assert self.out_shape[1] == 1, \
			"out_height must be 1, convolution only occurs across temporal dimension"

		modules = nn.ModuleList()
		modules.append(nn.ReplicationPad2d((pad_l, pad_r, 0, 0))) # XXX Reflection or Circular Pad?
		modules.append(
			init_layer(
				nn.utils.weight_norm(nn.Conv2d(self.in_shape[0], self.out_shape[0],
					kernel_size=(self.in_shape[1], kernel_size), stride=1,
					dilation=(1, dilation), groups=1, bias=True)),
				act=act,
				init_method=init
			)
		)
		modules.append(PYTORCH_ACT_MAPPING.get(act)())
		modules.append(nn.AlphaDropout(dropout) if (act in ('selu',)) \
			else nn.Dropout(dropout))
		self.layer = nn.Sequential(*modules)

	@classmethod
	def get_out_height(cls, in_height):
		return 1

	@classmethod
	def get_out_width(cls, in_width, kernel_size, dilation=1, padding_size=0, \
		stride=1):
		return int(np.floor(
			((in_width+padding_size-dilation*(kernel_size-1)-1)/stride)+1
		).item(0))

	def forward(self, x):
		return self.layer(x)

class ResidualBlock(nn.Module):
	"""
	Residual Block Module
	Wraps a residual connection around a network.
                       ____________________
	         ______|__net.forward(x)__|______
	         |                              |
	x -------|     ____________________     +-------> residual_block(x)
	         |_____|__downsample(x)___|_____|

	This module will propagate the input through the network and add the result
	to the input. The input might need to be downsampled to facillitate the addition.
	"""
	def __init__(self, net, act, downsample_type='linear', init='xavier_uniform', \
		use_residual=True):
		"""
		Args:
			net (nn.Module): nn.Module to wrap
			act (str): activation function
			downsample_type (str): method of downsampling used by residual block
			init (str): layer weight initialization method
		"""
		super().__init__()
		assert_has_shape_attr(net)
		self.in_shape = net.in_shape
		self.out_shape = net.out_shape
		self.net = net
		self.use_residual = use_residual
		if (self.use_residual):
			self.out_act = PYTORCH_ACT_MAPPING.get(act)()
			self.downsample = None
			if (self.net.in_shape != self.net.out_shape):
				if (downsample_type == 'linear'):
					self.padding = lambda x: x
					self.downsample = init_layer(nn.Linear(self.net.in_shape[0],
						self.net.out_shape[0], bias=True), act=act, init_method=init)
				elif (downsample_type == 'conv1d'):
					raise NotImplementedError()
				elif (downsample_type == 'conv2d'):
					padding_size = max(self.net.out_shape[2] - self.net.in_shape[2], 0)
					pad_l = padding_size//2
					pad_r = padding_size - pad_l
					self.padding = nn.ZeroPad2d((pad_l, pad_r, 0, 0)) # XXX Reflection or Circular Pad?
					self.downsample = init_layer(
						nn.Conv2d(self.net.in_shape[0], self.net.out_shape[0],
						kernel_size=(self.net.in_shape[1], 1), bias=True),
						act=act, init_method=init)

	def forward(self, x):
		net_out = self.net(x)

		if (self.use_residual):
			residual = x if (isnt(self.downsample)) else self.downsample(self.padding(x))
			try:
				return self.out_act(net_out + residual)
			except RuntimeError as e:
				print(e)
				logging.error('net(x).shape:   {}'.format(self.net(x).shape))
				logging.error('residual.shape: {}'.format(residual.shape))
				sys.exit(0)
		else:
			return net_out


# ********** MODEL MODULES **********
class TemporalConvNet(nn.Module):
	"""
	Temporal Convolutional Network

	The network performs 2D convolutions but only over the temporal (width) dimension.
	All kernel size, dilation, and padding size parameters only affect the temporal
	dimension of the kernel or input. For more information see TemporalLayer2d.

	Note: Receptive Field Size = Number TCN Blocks * Kernel Size * Last Layer Dilation Size
	"""
	def __init__(self, in_shape, block_channels=[[5, 5, 5]], num_blocks=1,
		kernel_sizes=3, dropouts=0.0, global_dropout=.5, global_dilation=True,
		block_act='elu', out_act='relu', block_init='xavier_uniform',
		out_init='xavier_uniform', pad_mode='same', downsample_type='conv2d',
		ob_out_shapes=None, ob_act='linear', ob_init='xavier_uniform'):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor,
				expects a shape (in_channels, in_height, in_width)
			block_channels (list * list): cnn channel sizes in each block,
				or individual channel sizes per block in sequence
			num_blocks (int): number of residual blocks,
				each block consists of a tcn network and residual connection
			kernel_sizes (int|list): list of CNN kernel sizes (across width dimension only),
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
			pad_mode('same'|'full'): padding method to use
			downsample_type (str): method of downsampling used by residual block
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
				padding_size = get_padding(pad_mode, layer_in_shape[2], kernel_size,
					dilation=dilation)
				out_height = TemporalLayer2d.get_out_height(layer_in_shape[1])
				out_width = TemporalLayer2d.get_out_width(layer_in_shape[2],
					kernel_size=kernel_size, dilation=dilation, padding_size=padding_size)
				layer_out_shape = (out_channels, out_height, out_width)
				dropout = global_dropout \
					if (isnt(dropouts) or i >= len(dropouts) or isnt(dropouts[i])) \
					else dropouts[i]
				layer = TemporalLayer2d(in_shape=layer_in_shape, out_shape=layer_out_shape,
					act=block_act, kernel_size=kernel_size, padding_size=padding_size,
					dilation=dilation, dropout=dropout, init=block_init)
				layers.append((f'tl[{layer_in_shape}->{layer_out_shape}]_{b}_{l}', layer))
				layer_in_shape = layer_out_shape
				i += 1
			block_out_shape = layer_out_shape
			net = nn.Sequential(OrderedDict(layers))
			net.in_shape, net.out_shape = block_in_shape, block_out_shape
			#net.in_shape, net.out_shape = layers[0][1].in_shape, layers[-1][1].out_shape
			blocks.append((f'rb[{downsample_type}]_{b}',
				ResidualBlock(net, out_act, downsample_type=downsample_type, init=out_init)
			))
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
	Stacked Temporal Convolutional Network
	Creates a TCN with fixed size output channels for each convolutional layer.
	For more information see TemporalConvNet.
	"""
	def __init__(self, in_shape, size=128, depth=3, kernel_sizes=3,
		input_dropout=0.0, output_dropout=0.0, global_dropout=.5,
		global_dilation=True, block_act='elu', out_act='relu',
		block_init='xavier_uniform', out_init='xavier_uniform',
		pad_mode='full', downsample_type='conv2d',
		ob_out_shapes=None, ob_act='linear', ob_init='xavier_uniform'):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor,
				expects a shape (in_channels, in_height, in_width)
			size (int): network embedding size
			depth (int): number of hidden layers to stack
			kernel_sizes (int|list): list of CNN kernel sizes (across width dimension only),
				if a list its length must be either 1 or len(block_channels)
			input_dropout (float): first layer dropout
			output_dropout (float): last layer dropout
			global_dropout (float): default dropout probability
			global_dilation (bool): whether to use global or block indexed dilation
			block_act (str): activation function of each layer in each block
			out_act (str): output activation of each block
			block_init (str): hidden layer weight initialization method
			out_init (str): output layer weight initialization method
			pad_mode('same'|'full'): padding method to use
			downsample_type (str): method of downsampling used by residual block
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
			out_init=out_init, pad_mode=pad_mode, downsample_type=downsample_type,
			ob_out_shapes=ob_out_shapes, ob_act=ob_act, ob_init=ob_init)

	@classmethod
	def suggest_params(cls, trial=None, num_classes=2, add_ob=False):
		"""
		suggest model hyperparameters from an optuna trial object
		or return fixed default hyperparameters
		"""
		if (is_valid(trial)):
			params = {
				'size': trial.suggest_int('size', 1, 32),
				'depth': trial.suggest_int('depth', 2, 5),
				'kernel_sizes': trial.suggest_categorical('kernel_sizes', \
					(7, 9, 15, 17, 23, 25, 31, 33, 39, 41, 47, 49)),
					# (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33)),
				# 'kernel_sizes': trial.suggest_int('kernel_sizes', 3, 33, step=2),
				'input_dropout': trial.suggest_float('input_dropout', \
					0.0, 1.0, step=1e-2),
				'output_dropout': trial.suggest_float('output_dropout', \
					0.0, 1.0, step=1e-2),
				'global_dropout': trial.suggest_float('global_dropout', \
					0.0, 1.0, step=1e-2),
				'global_dilation': True,
				'block_act': 'relu', 
				'out_act': 'relu',
				'block_init': 'kaiming_uniform',
				'out_init': 'kaiming_uniform',
				'pad_mode': 'full',
				'downsample_type': 'conv2d',
				'label_size': num_classes-1,
				'ob_out_shapes': num_classes if (add_ob) else None,
				'ob_act': 'relu',
				'ob_init': 'kaiming_uniform' # 'xavier_uniform'
			}
			# params = {
			# 	'size': trial.suggest_int('size', 2**5, 2**8, step=8),
			# 	'depth': trial.suggest_int('depth', 1, 5),
			# 	# 'kernel_sizes': trial.suggest_int('kernel_sizes', 3, 15, step=1),
			# 	'kernel_sizes': trial.suggest_categorical('kernel_sizes', \
			# 		(4, 8, 16, 24)),
			# 	'input_dropout': trial.suggest_float('input_dropout', \
			# 		0.0, 1.0, step=1e-6),
			# 	'output_dropout': trial.suggest_float('output_dropout', \
			# 		0.0, 1.0, step=1e-6),
			# 	'global_dropout': trial.suggest_float('global_dropout', \
			# 		0.0, 1.0, step=1e-6),
			# 	'global_dilation': True,
			# 	'block_act': trial.suggest_categorical('block_act', \
			# 		PYTORCH_ACT1D_LIST[4:]),
			# 	'out_act': trial.suggest_categorical('out_act', \
			# 		PYTORCH_ACT1D_LIST[4:]),
			# 	'block_init': trial.suggest_categorical('block_init', \
			# 		PYTORCH_INIT_LIST[2:]),
			# 	'out_init': trial.suggest_categorical('out_init', \
			# 		PYTORCH_INIT_LIST[2:]),
			# 	'pad_mode': trial.suggest_categorical('pad_mode', \
			# 		('same', 'full')),
			# 	'downsample_type': 'conv2d',
			# 	'label_size': num_classes-1,
			# 	'ob_out_shapes': num_classes if (add_ob) else None,
			# 	'ob_act': trial.suggest_categorical('ob_act', \
			# 		PYTORCH_ACT1D_LIST[4:-1]),
			# 	'ob_init': trial.suggest_categorical('ob_init', \
			# 		PYTORCH_INIT_LIST[2:])
			# }
		else:
			params = {
				'size': 128,
				'depth': 3,
				'kernel_sizes': 8,
				'input_dropout': 0.0,
				'output_dropout': 0.0,
				'global_dropout': .5,
				'global_dilation': True,
				'block_act': 'elu',
				'out_act': 'relu',
				'block_init': 'xavier_uniform',
				'out_init': 'xavier_uniform',
				'pad_mode': 'full',
				'downsample_type': 'conv2d',
				'label_size': num_classes-1,
				'ob_out_shapes': num_classes if (add_ob) else None,
				'ob_act': 'linear',
				'ob_init': 'xavier_uniform'
			}
		return params

class TransposedTCN(nn.Module):
	"""
	Single block 2D TCN where dims (by default the first two) are transposed between convolutions.
	"""
	def __init__(self, in_shape, channels=[3, 1], kernel_sizes=3,
		input_dropout=0.0, output_dropout=0.0, global_dropout=0.0,
		use_dilation=True, block_act='relu', out_act='relu',
		block_init='kaiming_uniform', out_init='kaiming_uniform',
		pad_mode='full', tdims=(2, 1), use_residual=True, downsample_type='conv2d',
		ob_out_shapes=None, ob_act='linear', ob_init='xavier_uniform'):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor,
				expects a shape (in_channels, in_height, in_width)
			channels (list): cnn channel sizes
			kernel_sizes (int|list): list of CNN kernel sizes (across width dimension only),
				if a list its length must be either 1 or len(channels)
			input_dropout (float): first layer dropout
			output_dropout (float): last layer dropout
			global_dropout (float): default dropout probability
			use_dilation (bool): whether to use dilation
			block_act (str): activation function of each layer in each block
			out_act (str): output activation of each block
			block_init (str): hidden layer weight initialization method
			out_init (str): output layer weight initialization method
			pad_mode ('same'|'full'): padding method to use
			tdims (tuple): dimensions to transpose (batch index included) 
			use_residual(): 
			downsample_type (str): method of downsampling used by residual block
			ob_out_shapes
			ob_act
			ob_ini
		"""
		super().__init__()
		self.in_shape = layer_in_shape = in_shape
		dropouts = [global_dropout] * len(channels)
		dropouts[0], dropouts[-1] = input_dropout, output_dropout
		kernel_sizes = list_wrap(kernel_sizes)
		layers = []

		for l, out_channels in enumerate(channels):
			dilation = 2**l if (use_dilation) else 1
			kernel_size = kernel_sizes[l%len(kernel_sizes)]
			dropout = dropouts[l%len(dropouts)]
			padding_size = get_padding(pad_mode, layer_in_shape[2], kernel_size,
				dilation=dilation)
			out_height = TemporalLayer2d.get_out_height(layer_in_shape[1])
			out_width = TemporalLayer2d.get_out_width(layer_in_shape[2],
				kernel_size=kernel_size, dilation=dilation, padding_size=padding_size)

			# TCN layer
			layer_out_shape = (out_channels, out_height, out_width)
			layer = TemporalLayer2d(in_shape=layer_in_shape, out_shape=layer_out_shape,
				act=block_act, kernel_size=kernel_size, padding_size=padding_size,
				dilation=dilation, dropout=dropout, init=block_init)
			layers.append((f'tl[{layer_in_shape}->{layer_out_shape}]_0_{l}', layer))

			if (l < len(channels)-1):
				# Transpose before next TCN layer
				trp = Apply(partial(torch.transpose, dim0=tdims[0], dim1=tdims[1]))
				layer_in_shape = (layer_out_shape[1], layer_out_shape[0], layer_out_shape[2])
				layers.append((f'transpose{tdims}[{layer_out_shape}->{layer_in_shape}]', trp))

		net = nn.Sequential(OrderedDict(layers))
		self.out_shape = layer_out_shape
		net.in_shape, net.out_shape = self.in_shape, self.out_shape
		self.model = ResidualBlock(net, out_act, downsample_type=downsample_type, init=out_init, use_residual=use_residual)

		# GenericModel uses these params to to append an OutputBlock to the model
		self.ob_out_shapes = ob_out_shapes	# If this is None, no OutputBlock is added
		self.ob_act, self.ob_init = ob_act, ob_init

	def forward(self, x):
		return self.model(x)

class FFN(nn.Module):
	"""
	MLP Feedforward Network
	"""
	def __init__(self, in_shape, out_shapes=[128, 128, 128], act='relu',
		init='xavier_uniform'):
		super().__init__()
		io_shapes = [np.product(in_shape).item(0), *out_shapes]
		ffn_layers = [('flatten', nn.Flatten(start_dim=1, end_dim=-1))]
		self.in_shape, self.out_shape = in_shape, (io_shapes[-1],)

		for l, (i, o) in enumerate(pairwise(io_shapes)):
			ffn_layers.append((f'ff_{l}', init_layer(nn.Linear(i, o), act=act, \
				init_method=init)))
			if (act not in ('linear', None)):
				ffn_layers.append((f'af_{l}', PYTORCH_ACT_MAPPING.get(act)()))

		self.model = nn.Sequential(OrderedDict(ffn_layers))

	def forward(self, x):
		return self.model(x)

	@classmethod
	def suggest_params(cls, trial=None, num_classes=2, add_ob=False):
		"""
		suggest model hyperparameters from an optuna trial object
		or return fixed default hyperparameters
		"""
		if (is_valid(trial)):
			size = trial.suggest_int('size', 2**5, 2**8, step=8),
			depth = trial.suggest_int('depth', 1, 5)

			params = {
				'out_shapes': [size] * depth,
				'act': trial.suggest_categorical('block_act', \
					PYTORCH_ACT1D_LIST[4:]),
				'init': trial.suggest_categorical('block_init', \
					PYTORCH_INIT_LIST[2:]),
				'label_size': num_classes-1,
			}
		else:
			params = {
				'out_shapes': [32, 32, 32],
				'act': 'relu',
				'init': 'xavier_uniform',
				'label_size': num_classes-1,
			}
		return params

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


MODEL_MAPPING = {
	'ffn': FFN,
	'stcn': StackedTCN,
	'ttcn': TransposedTCN
}

