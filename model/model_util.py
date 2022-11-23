import sys
import os
import logging
from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# from kymatio.torch import Scattering1D as pt_wavelet_scatter_1d

from common_util import is_type, is_valid, isnt, list_wrap, assert_has_all_attr, pairwise, odd_only
from model.common import PYTORCH_ACT_MAPPING, PYTORCH_ACT1D_LIST, PYTORCH_INIT_LIST


# ********** HELPER FUNCTIONS **********
def squeeze_shape(shape):
	"""
	Return shape tuple with 1's removed
	"""
	return tuple(filter(lambda x: x!=1, shape))

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
			gain = nn.init.calculate_gain(act or 'linear')
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
	raise DeprecationWarning('use MHA module instead')
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

# class WaveletScatter1d(nn.Module):
# 	"""
# 	Apply One Dimensional Wavelet Scattering Transform on a vector and return the coefficients as a tuple of torch tensors.
# 	Wrapper around the Kymatio library, uses analytic (complex-valued) Morlet Wavelets down to 2nd order wavelets.
# 	arXiv:1812.11214
# 	"""
# 	def __init__(self, input_shape, max_scale_power, fo_wavelets):
# 		"""
# 		Args:
# 			input_shape (tuple): Length of the input vector
# 			max_scale_power (int): Maximum base 2 log scale of scattering transform
# 			fo_wavelets (int>=1): Number of first order wavelets per octave (2nd order is fixed to 1)
# 		"""
# 		super().__init__()
# 		self.scatter = pt_wavelet_scatter_1d(max_scale_power, input_shape, fo_wavelets)
# 		#self.scatter.cuda()
# 		meta = self.scatter.meta()
# 		self.orders = [np.where(meta['order'] == i) for i in range(3)]

# 	def forward(self, x):
# 		Sx = self.scatter(x)
# 		coefficients = tuple(Sx[order] for order in self.orders)
# 		return coefficients

class Apply(nn.Module):
	"""
	Module wrapper to apply a function to the input.
	"""
	def __init__(self, fn):
		super().__init__()
		self.fn = fn

	def forward(self, x):
		return self.fn(x)

class SwapLinear(nn.Module):
	"""
	Linear layer applied chosen dimension
	"""
	def __init__(self, in_features, out_features, bias=True, dim=-1):
		super().__init__()
		self.lin = nn.Linear(in_features, out_features, bias=True)
		self.dim = dim

	def forward(self, x):
		return self.lin(x.transpose(self.dim, -1)).transpose(self.dim, -1)

class TransposeModule(nn.Module):
	def __init__(self, dim0, dim1):
		super().__init__()
		self.dim0, self.dim1 = dim0, dim1

	def forward(self, x):
		return x.transpose(self.dim0, self.dim1)


# ********** BLOCK MODULES **********
class OutputBlock(nn.Module):
	"""
	Appends a feedforward layer or block to the end of an arbitrary embedding network.
	Used for Classification or Regression depending on the loss function used.
	"""
	def __init__(self, emb, ob_name, **ob_kwargs):
		"""
		Args:
			emb (nn.Module): embedding network to add output layer to
			ob_name:
		"""
		super().__init__()
		assert_has_shape_attr(emb)
		self.emb = emb
		self.out = MODEL_MAPPING[ob_name](self.emb.out_shape, **ob_kwargs)
		self.in_shape = self.emb.in_shape
		self.out_shape = self.out.out_shape

	def forward(self, x):
		return self.out(self.emb(x))

	@classmethod
	def wrap(cls, emb):
		"""
		Wrap/append an OutputBlock to the passed embedding model
		if ob_out_shapes is not None and return original model otherwise
		"""
		model = emb
		if (hasattr(model, 'ob_name') and hasattr(model, 'ob_params') and is_valid(model.ob_name)):
			ob_params = model.ob_params or {}
			model = OutputBlock(model, model.ob_name, **ob_params)
		return model

class MHA(nn.Module):
	"""
	Stacked Multihead Attention module (uses torch.nn.MultiheadAttention)

	This module inputs/outputs a tensor(s) shaped like (n, {S, L}, E[{q,k,v}]), where:
		* n: batch
		* {S, L}: {source, target} set
		* E[{q, k, v}]: {query, key, value} embedding
	"""
	def __init__(self, in_shape, num_heads=1, dropout=0.0, depth=1, kdim=None, vdim=None):
		"""
		Args:
			in_shape (tuple): (L, E[q])
			num_heads (int>0): num attention heads
			dropout (float>=0): dropout
			depth (int>0): network depth
			kdim: E[k] (if None, assumes E[k]==E[q])
			vdim: E[v] (if None, assumes E[v]==E[q])
		"""
		super().__init__()
		self.in_shape, self.out_shape = in_shape, in_shape
		self.embed_dim = in_shape[-1]
		self.num_heads = num_heads
		self.mhas = nn.ModuleList([nn.MultiheadAttention(
			self.embed_dim, self.num_heads, dropout=dropout, bias=True, add_bias_kv=False,
			add_zero_attn=False, kdim=kdim, vdim=vdim, batch_first=True) for _ in range(depth)])

	def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
		"""
		Applies Pytorch Multiheaded Attention

		Attention(Q, K, V) = softmax(QK'/sqrt(d_k)) V

		MH(Q, K, V) = Concat[head1; head2; ...; headn] Wo

			where head[i] = Attention(Q Wq[i], K Wk[i], V Wv[i])

		Args:
			mha (torch.nn.modules.activation.MultiheadAttention): pytorch Multihead attention object
			q (torch.tensor): query tensor (n, L, E[q])
			k (torch.tensor): key tensor (n, S, E[k]) (if None, k=q)
			v (torch.tensor): value tensor (n, S, E[v]) (if None, v=q)

		Returns:
			torch.tensor shaped like (n, S, E[q])
		"""
		_k = k if (is_valid(k)) else q
		_v = v if (is_valid(v)) else q

		for i, mha in enumerate(self.mhas):
			q, _ = mha(q, _k, _v, key_padding_mask=key_padding_mask,
				need_weights=True, attn_mask=attn_mask)

		return q.contiguous()

class LaplaceAttention(nn.Module):
	"""
	Laplace Exponential Attention module

	This module inputs/outputs a tensor shaped like (N, C, S),
	where:
		* N: batch
		* C: channel/embedding
		* S: sequence

	act(sum_i(-l1(k - q) / scale))
	Adapted from: https://github.com/deepmind/neural-processes/blob/master/attentive_neural_process.ipynb
	"""
	def __init__(self, in_shape, scale=2, act='smax'):
		"""
		Args:
			in_shape (tuple): (C, S)
			scale (float): L1 distance scale
		"""
		super().__init__()
		self.in_shape, self.out_shape = in_shape, in_shape
		self.scale = scale
		self.out_act = (af := PYTORCH_ACT_MAPPING.get(act, None)) and af()

	def forward(self, q, k=None, v=None):
		if (isnt(k)):
			# self attention -> use reversed sequence queries as keys
			k = torch.flip(q, (-1, -2))
		k = k.unsqueeze(1)				# [n, C, S] -> [n, 1, C, S]
		v = v if (is_valid(v)) else q			# [n, C, S_v]
		q = q.unsqueeze(2)				# [n, C, S] -> [n, C, 1, S]
		try:
			weights = - (torch.abs(k - q)/self.scale)	# - scaled L1 -> [n, C, C, S]
			weights = weights.sum(dim=-1)			# [n, C, C, S] -> [n, C, C]
			if (is_valid(self.out_act)):
				weights = self.out_act(weights)
		except Exception as err:
			print("Error! model_util.py > LaplaceAttention > forward() > ...\n",
				sys.exc_info()[0], err)
			print(f'{q.shape=}')
			print(f'{k.shape=}')
			print(f'{v.shape=}')
			raise err

		try:
			rep = torch.einsum('ncd,nds->ncs', weights, v)	# reweight values -> [n, C, S_v]
		except Exception as err:
			print("Error! model_util.py > LaplaceAttention > forward() > torch.einsum()\n",
				sys.exc_info()[0], err)
			print(f'{weights.shape=}')
			print(f'{q.shape=}')
			print(f'{k.shape=}')
			print(f'{v.shape=}')
			raise err
		return rep

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
		dilation, dropout, dropout_type='2d', init='xavier_uniform'):
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
		# modules.append(nn.ReplicationPad2d((pad_l, pad_r, 0, 0)))
		modules.append(nn.ZeroPad2d((pad_l, pad_r, 0, 0)))
		modules.append(
				nn.utils.weight_norm(init_layer(
					nn.Conv2d(self.in_shape[0], self.out_shape[0],
						kernel_size=(self.in_shape[1], kernel_size), stride=1,
						dilation=(1, dilation), groups=1, bias=True),
					act=act,
					init_method=init
				),
			)
		)
		if (is_valid(act_fn := PYTORCH_ACT_MAPPING.get(act, None))):
			modules.append(act_fn())
		if (is_valid(dropout)):
			if (dropout_type == '2d'):
				drop = nn.Dropout2d(dropout)
			elif (act in ('selu',)):
				drop = nn.AlphaDropout(dropout)
			else:
				drop = nn.Dropout(dropout)
			modules.append(drop)
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
			self.out_act = (af := PYTORCH_ACT_MAPPING.get(act, None)) and af()
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

					# self.padding = nn.ReplicationPad2d((pad_l, pad_r, 0, 0))
					self.padding = nn.ZeroPad2d((pad_l, pad_r, 0, 0))
					self.downsample = init_layer(
						nn.Conv2d(self.net.in_shape[0], self.net.out_shape[0],
						kernel_size=(self.net.in_shape[1], 1), bias=True),
						act=act, init_method=init)

	def forward(self, x):
		net_out = self.net(x)

		if (self.use_residual):
			residual = x if (isnt(self.downsample)) else self.downsample(self.padding(x))
			try:
				res = net_out + residual
				if (is_valid(self.out_act)):
					res = self.out_act(res)
				return res
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

	The network performs 2D convolutions but only over the temporal (W) dimension.
	All kernel size, dilation, and padding size parameters only affect the temporal
	dimension of the kernel or input. For more information see TemporalLayer2d.

	note: nth layer receptive field, r_n = r[n-1] + (kernel_size[n] - 1) * dilation[n]
	"""
	def __init__(self, in_shape, block_channels=[[8, 8], [4, 4], [1, 1]],
		kernel_sizes=3, dropouts=0.0, dropout_type='2d', dilation_factor=2,
		block_act='relu', out_act='relu', block_init='xavier_uniform',
		out_init='xavier_uniform', pad_mode='same', downsample_type='conv2d',
		ob_name=None, ob_params=None):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor,
				expects a shape (in_channels, in_height, in_width)
			block_channels (list * list): channel size topology
			kernel_sizes (list): list of CNN kernel sizes (across width dimension only)
			dropouts (float|list): dropout probability by block
				if an element is None, disables dropout at that block
			block_act (str): activation function of each layer in each block
			out_act (str): output activation of each block
			block_init (str): hidden layer weight initialization method
			out_init (str): output layer weight initialization method
			pad_mode('same'|'full'): padding method to use
			downsample_type (str): method of downsampling used by residual block
			ob_name
			ob_params
			"""
		super().__init__()
		self.in_shape = block_in_shape = in_shape
		assert len(block_channels) == len(kernel_sizes) == len(dropouts)
		assert is_type(dilation_factor, int) and dilation_factor > 0
		blocks = []

		for b, block_channel in enumerate(block_channels):
			layer_in_shape = block_in_shape
			k = kernel_sizes[b]
			do = dropouts[b]
			di = dilation_factor**b
			layers = []

			for l, layer_channel in enumerate(block_channel):
				padding_size = get_padding(pad_mode, layer_in_shape[2], k, dilation=di)
				out_height = TemporalLayer2d.get_out_height(layer_in_shape[1])
				out_width = TemporalLayer2d.get_out_width(layer_in_shape[2], kernel_size=k, dilation=di, padding_size=padding_size)
				layer_out_shape = (layer_channel, out_height, out_width)
				layer = TemporalLayer2d(in_shape=layer_in_shape, out_shape=layer_out_shape,
					act=block_act, kernel_size=k, padding_size=padding_size,
					dilation=di, dropout=do, dropout_type=dropout_type, init=block_init)
				layers.append((f'tl[{layer_in_shape}->{layer_out_shape}]_{b}_{l}', layer))
				layer_in_shape = layer_out_shape
			block_out_shape = layer_out_shape
			net = nn.Sequential(OrderedDict(layers))
			net.in_shape, net.out_shape = block_in_shape, block_out_shape
			#net.in_shape, net.out_shape = layers[0][1].in_shape, layers[-1][1].out_shape
			blocks.append((f'rb[{downsample_type}]_{b}',
				ResidualBlock(net, out_act, downsample_type=downsample_type, init=out_init)
			))
			block_in_shape = block_out_shape
		self.out_shape = block_out_shape

		model = nn.Sequential(OrderedDict(blocks))
		model.in_shape = self.in_shape
		model.out_shape = self.out_shape
		model.ob_name = ob_name
		model.ob_params = ob_params
		self.model = OutputBlock.wrap(model)
		self.in_shape = self.model.in_shape
		self.out_shape = self.model.out_shape

	def forward(self, x):
		return self.model(x)

class StackedTCN(TemporalConvNet):
	"""
	Stacked Temporal Convolutional Network
	Creates a TCN with fixed size output channels for each convolutional layer.
	For more information see TemporalConvNet.
	"""
	def __init__(self, in_shape, size=8, depth=1, subdepth=2, collapse_out=False, kernel_sizes=3,
		input_dropout=None, output_dropout=None, global_dropout=None, dropout_type='2d',
		dilation_factor=2, block_act='relu', out_act='relu',
		block_init='xavier_uniform', out_init='xavier_uniform',
		pad_mode='full', downsample_type='conv2d',
		ob_name=None, ob_params=None):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor,
				expects a shape (in_channels, in_height, in_width)
			size (int): network embedding size
			depth (int): number of blocks to stack (not including collapse_out)
			subdepth (int): number of convolutions per block (convolutions between residual
				skip connections)
			collapse_out (bool): whether to add a tcn block of size `1` to the end to collapse the channels.
			kernel_sizes (int|list): list of CNN kernel sizes (across width dimension only),
				if a list its length must be depth
			input_dropout (float): first layer dropout
			output_dropout (float): last layer dropout
			global_dropout (float): default dropout probability
			block_act (str): activation function of each layer in each block
			out_act (str): output activation of each block
			block_init (str): hidden layer weight initialization method
			out_init (str): output layer weight initialization method
			pad_mode('same'|'full'): padding method to use
			downsample_type (str): method of downsampling used by residual block
			ob_name
			ob_params
		"""
		block_channels = [[size] * subdepth] * depth
		dropouts = [global_dropout] * depth
		if (is_type(kernel_sizes, int)):
			kernel_sizes = [kernel_sizes] * depth
		if (collapse_out):
			block_channels.append([1] * subdepth)
			dropouts.append(global_dropout)
			kernel_sizes.append(kernel_sizes[-1])
		dropouts[0], dropouts[-1] = input_dropout, output_dropout
		
		super().__init__(in_shape, block_channels=block_channels,
			kernel_sizes=kernel_sizes, dropouts=dropouts, dropout_type=dropout_type,
			dilation_factor=dilation_factor,
			block_act=block_act, out_act=out_act,
			block_init=block_init, out_init=out_init, pad_mode=pad_mode,
			downsample_type=downsample_type, ob_name=ob_name, ob_params=ob_params)

	@classmethod
	def get_receptive_field(cls, depth, kernel_size, dilation_factor):
		"""
		note: nth layer receptive field (for constant kernel size 'k', global dilation factor 'f')
		Keep in mind here dilation factor isn't the actual dilation but the factor by which
		dilation grows each layer.
			          n
		r_n = (k-1) * ∑ (f ** i)
			          0
		"""
		return (kernel_size-1) * sum(dilation_factor**i for i in range(depth))

class TransposedTCN(nn.Module):
	"""
	Single block 2D TCN where dims (by default the first two) are transposed between convolutions.
	"""
	def __init__(self, in_shape, channels=[3, 1], kernel_sizes=3,
		input_dropout=None, output_dropout=None, global_dropout=None,
		use_dilation=True, block_act='relu', out_act='relu',
		block_init='kaiming_uniform', out_init='kaiming_uniform',
		pad_mode='full', tdims=(2, 1), use_residual=True, downsample_type='conv2d',
		ob_name=None, ob_params=None):
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
			ob_params
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

		model = ResidualBlock(net, out_act, downsample_type=downsample_type, init=out_init, use_residual=use_residual)
		model.in_shape = self.in_shape
		model.out_shape = self.out_shape
		model.ob_name = ob_name
		model.ob_params = ob_params
		self.model = OutputBlock.wrap(model)
		self.in_shape = self.model.in_shape
		self.out_shape = self.model.out_shape

	def forward(self, x):
		return self.model(x)

class FFN(nn.Module):
	"""
	MLP Feedforward Network
	"""
	def __init__(self, in_shape, out_shapes=[128,],
		flatten=False, flatten_start=1,
		input_dropout=None, output_dropout=None, global_dropout=None,
		act='relu', act_output=True, init='xavier_uniform'):
		super().__init__()
		self.in_shape, self.out_shape = in_shape, (out_shapes[-1],)
		if (flatten):
			ffn_layers = [('flatten', nn.Flatten(start_dim=flatten_start, end_dim=-1))]
			ins = np.product(self.in_shape).item(0)
		else:
			ffn_layers = []
			ins = self.in_shape[-1] # treat previous dims as batch dimensions
		io_shapes = [ins, *out_shapes]
		dropouts = [global_dropout] * len(out_shapes)
		dropouts[0], dropouts[-1] = input_dropout, output_dropout

		for l, (i, o) in enumerate(pairwise(io_shapes)):
			lay = init_layer(nn.Linear(i, o), act=act, init_method=init)
			ffn_layers.append((f'ff_{l}', lay))
			if (act not in ('linear', None) and (l < len(out_shapes)-1 or act_output)):
				ffn_layers.append((f'af_{l}', PYTORCH_ACT_MAPPING.get(act)()))
			if (is_valid(do := dropouts[l])):
				ffn_layers.append((f'do_{l}', nn.Dropout(do)))

		self.model = nn.Sequential(OrderedDict(ffn_layers))

	def forward(self, x):
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

# ********** NORMALIZATION MODULES **********
class InstanceNorm15d(nn.Module):
	"""
	Run 1D instance norm over W dimension of input shaped (.., C, H, W), where
		C -> nchan, creates a separate norm module per channel
		H -> nser
		W -> series dimension to normalize over
	Valid input shapes:
		* (n, C, H, W)
		* (n, e, C, H, W)
	"""
	def __init__(self, num_channels, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, to_cuda=True, **params):
		super().__init__()
		self.nchan = num_channels
		self.nser = num_features
		self.norm1d = [nn.InstanceNorm1d(self.nser, eps=eps, momentum=momentum,
			affine=affine, track_running_stats=track_running_stats, **params)
			for i in range(self.nchan)]
		if (to_cuda):
			self.norm1d = [s.cuda() for s in self.norm1d]

	def forward(self, x):
		if (x.ndim == 4):
			# use dim=0 as batch dimension for InstanceNorm1d operation, [n, H, W]:
			chan_out = [self.norm1d[c](x[:, c]).unsqueeze(1) for c in range(self.nchan)]
			out = torch.cat(chan_out, dim=1)
		elif (x.ndim == 5):
			batch_out = []
			for b in range(x.shape[0]):
				# use dim=1 as batch dimension for InstanceNorm1d operation, [o, H, W]:
				chan_out = [self.norm1d[c](x[b, :, c]).unsqueeze(1) for c in range(self.nchan)]
				batch_out.append(torch.cat(chan_out, dim=1).unsqueeze(0))
			out = torch.cat(batch_out, dim=0)
		return out

class BatchNorm15d(nn.Module):
	"""
	Run 1D batch norm over (bdim, W) dimensions of input shaped (.., C, H, W), where
		C -> nchan, creates a separate norm module per channel
		H -> nser
		W -> series dimension to normalize over
	Valid input shapes:
		* (n, C, H, W)
		* (n, e, C, H, W)
	Setting bdim determines the batch dim for batch norm (only used if ndim > 4).
	"""
	def __init__(self, num_channels, num_features, bdim=1, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, to_cuda=True, **params):
		super().__init__()
		self.nchan = num_channels
		self.nser = num_features
		self.bdim = bdim
		self.norm1d = [nn.BatchNorm1d(self.nser, eps=eps, momentum=momentum,
			affine=affine, track_running_stats=track_running_stats, **params)
			for i in range(self.nchan)]
		if (to_cuda):
			self.norm1d = [s.cuda() for s in self.norm1d]

	def forward(self, x):
		raise NotImplementedError() # Need to test
		if (x.ndim == 4):
			# always use bdim=0 as batch dimension for InstanceNorm1d operation, [n, H, W]:
			chan_out = [self.norm1d[c](x[:, c]).unsqueeze(1) for c in range(self.nchan)]
			out = torch.cat(chan_out, dim=1)
		elif (x.ndim == 5):
			batch_out = []
			if (self.bdim == 0):
				for b in range(x.shape[1]):
					# use bdim=0 as batch dimension for InstanceNorm1d operation, [n, H, W]:
					chan_out = [self.norm1d[c](x[:, b, c]).unsqueeze(1) for c in range(self.nchan)]
					batch_out.append(torch.cat(chan_out, dim=1).unsqueeze(self.bdim))
			elif (self.bdim == 1):
				for b in range(x.shape[0]):
					# use bdim=1 as batch dimension for InstanceNorm1d operation, [o, H, W]:
					chan_out = [self.norm1d[c](x[b, :, c]).unsqueeze(1) for c in range(self.nchan)]
					batch_out.append(torch.cat(chan_out, dim=1).unsqueeze(self.bdim))
			out = torch.cat(batch_out, dim=self.bdim)
		return out


MODEL_MAPPING = {
	'ffn': FFN,
	'mha': MHA,
	'la': LaplaceAttention,
	'stcn': StackedTCN,
	'ttcn': TransposedTCN
}

NORM_MAPPING = {
	'gn': lambda in_shape, num_groups, **params: \
		nn.GroupNorm(num_groups=num_groups, num_channels=in_shape[0], **params),
	'ln': lambda in_shape, num_groups, **params: \
		nn.GroupNorm(num_groups=1, num_channels=in_shape[0], **params),
	'in1d': lambda in_shape, **params: \
		nn.InstanceNorm1d(num_features=in_shape[0], **params),
	'in15d': lambda in_shape, **params: \
		InstanceNorm15d(num_channels=in_shape[0], num_features=in_shape[1], **params),
	'in2d': lambda in_shape, **params: \
		nn.InstanceNorm2d(num_features=in_shape[0], **params),
	'in3d': lambda in_shape, **params: \
		nn.InstanceNorm3d(num_features=in_shape[0], **params),
	'bn1d': lambda in_shape, **params: \
		nn.BatchNorm1d(num_features=in_shape[0], **params),
	'bn15d': lambda in_shape, **params: \
		BatchNorm15d(num_channels=in_shape[0], num_features=in_shape[1], **params),
	'bn2d': lambda in_shape, **params: \
		nn.BatchNorm2d(num_features=in_shape[0], **params),
	'bn3d': lambda in_shape, **params: \
		nn.BatchNorm3d(num_features=in_shape[0], **params),
}

