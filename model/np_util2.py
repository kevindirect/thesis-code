"""
Neural Process utils
Kevin Patel
"""
import sys
import os
import math
from operator import mul
from functools import reduce, partial
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common_util import is_valid, isnt
from model.common import PYTORCH_ACT1D_LIST, PYTORCH_INIT_LIST
from model.model_util import log_prob_sigma, init_layer, get_padding, pt_multihead_attention, MODEL_MAPPING
# Tensors are column-major, shaped as (batch, channel, height, width) unless otherwise specified
# The in_shape and out_shape attributes of modules don't include the batch size
# Inspired by:
# * https://github.com/deepmind/neural-processes/blob/master/attentive_neural_process.ipynb
# * https://github.com/3springs/attentive-neural-processes/blob/master/neural_processes/models/neural_process/model.py
# * https://github.com/soobinseo/Attentive-Neural-Process/blob/master/module.py


# ********** HELPER FUNCTIONS **********
def pt_concat_xy(x, y, size=1, dim=2):
	assert(x.shape[0] == y.shape[0])
	xpd_y = y.reshape(y.shape[0], 1, 1, 1).float()

	if (dim == 1):
		xpd_y = xpd_y.expand(-1, size, *x.shape[2:])
	elif (dim == 2):
		xpd_y = xpd_y.expand(-1, x.shape[1], size, x.shape[3])
	elif (dim == 3):
		xpd_y = xpd_y.expand(-1, *x.shape[1:3], size)

	return torch.cat([x, xpd_y], dim=dim)

def get_xy_concat_dim(rt_name):
	return {
		'stcn': 2,
		'ffn': 3
	}.get(rt_name, 3)

def get_rt_in_shape(in_shape, label_size, cc_dim):
	rt_in_shape = list(in_shape)
	rt_in_shape[cc_dim-1] += label_size
	rt_in_shape.remove(1)
	return tuple(rt_in_shape)


# ********** HELPER MODULES **********
class DetEncoder(nn.Module):
	"""
	Attentive Neural Process (ANP) Deterministic Encoder.
	Takes in examples of shape (in_channels, in_height, in_width/sequence)

	Returns representation based on (Xc, yc) context points and Xt target points

	3. Transform (Xc, yc) -> V
	4. Run rep transform on V
	5. Run cross aggregation on Q, K, V to get final representation
		(cross attention: get weighted V based on similarity scores of Q to K)
	"""
	def __init__(self, in_shape, label_size, embed_size, class_agg=False,
		rt_name='mha', rt_params=None, xa_name='mha', xa_params=None):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor,
				expects a shape (in_channels, in_height, in_width)
			label_size (int>0): size of the label vector
			embed_size (int>0): query size (XXX remove param?)
			class_agg (bool): whether to aggregate across separate classes or globally
			rt_name (str): representation transform name
			rt_params (dict): representation transform hyperparameters
			xa_name (str): cross aggregation name
			xa_params (dict): cross aggregation hyperparameters
		"""
		super().__init__()
		self.in_shape = in_shape
		self.label_size = label_size
		if (isnt(rt_params)):
			rt_params = {}
		if (isnt(xa_params)):
			xa_params = {}

		self.cc_dim = get_xy_concat_dim(rt_name)
		rt_in_shape = get_rt_in_shape(self.in_shape, self.label_size, self.cc_dim)
		self.rep_transform = MODEL_MAPPING.get(rt_name, None)
		try:
			self.rep_transform = self.rep_transform and \
				self.rep_transform(rt_in_shape, **rt_params)
		except Exception as err:
			print("Error! np_util2.py > DetEncoder > __init__()\n",
				sys.exc_info()[0], err)
			print(f'{rt_in_shape=}')
			print(f'{rt_name=}')
			print(f'{rt_params=}')
			raise err

		xa_in_shape = self.rep_transform.out_shape if (is_valid(rt_name)) \
			else rt_in_shape
		self.cross_aggregation = MODEL_MAPPING[xa_name]
		try:
			self.cross_aggregation = self.cross_aggregation(xa_in_shape, **xa_params)
		except Exception as err:
			print("Error! np_util2.py > DetEncoder > __init__()\n",
				sys.exc_info()[0], err)
			print(f'{xa_in_shape=}')
			print(f'{xa_name=}')
			print(f'{xa_params=}')
			raise err

		self.class_agg = class_agg
		self.label_pad = nn.ConstantPad1d((0, self.label_size), 0.0) # padding needed because of label append
		self.out_shape = self.cross_aggregation.out_shape

	def forward(self, context_h, context_y, target_h):
		"""
		ANP Deterministic Encoder forward Pass
		Returns all representations, need to be aggregated to form r_*.
		* Include Positional Encoding?
		"""
		if (self.class_agg):
			raise NotImplementedError() # TODO
			classes = context_y.unique(sorted=True, return_inverse=False)
			# for each unique label value
			# 	get indices in context_y where equal to label value
			# 	index into context_h
			# 	self attention
			# 	append self attention outputs to list

			# print(context_h)
			# print(context_h.shape)
			for c in classes:
				idx = context_y == c

				# # add label to end of values
				# reps = pt_concat_xy(context_h[idx], context_y[idx], self.label_size, dim=self.cc_dim)
				# values = self.rep_transform(reps) if (self.rep_transform) else reps
				# queries, keys, values = target_h.squeeze(), self.label_pad(context_h[idx].squeeze()), values.squeeze()

				# # dont add label to end of values
				# reps = context_h[idx]
				# values = self.rep_transform(reps) if (self.rep_transform) else reps
				# queries, keys, values = target_h.squeeze(), context_h[idx].squeeze(), values.squeeze()
			sys.exit()

		else:
			queries, keys = target_h.squeeze(), self.label_pad(context_h.squeeze())
			reps = pt_concat_xy(context_h, context_y, self.label_size, dim=self.cc_dim) \
				.squeeze()
			values = self.rep_transform(reps) if (self.rep_transform) else reps
			queries = self.cross_aggregation(queries, keys, values)

		return queries


class LatEncoder(nn.Module):
	"""
	Attentive Neural Process (ANP) Latent Encoder.

	1. Transform (Xc, yc) -> enc
	2. Run self attention on enc
	3. Use aggregate representation (over sequence dim) of enc to parameterize a distribution (latent variable)
	* using linear layers for map_layer, alpha, and beta modules
	"""
	def __init__(self, in_shape, label_size, latent_size=256,
		rt_name='mha', rt_params=None, dist_type='normal',
		min_std=.01, use_lvar=False):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor,
				expects a shape (in_channels, in_height, in_width)
			label_size (int>0): size of the label vector
			latent_size (int>0): size of latent representation
			rt_name (str): representation transform name
			rt_params (dict): representation transform hyperparameters
			dist_type ('beta'|'normal'|'lognormal'): latent distribution
			min_std (float): value used to limit latent distribution sigma
			use_lvar (bool): whether to use log domain variance clipping
		"""
		super().__init__()
		self.in_shape = in_shape
		self.label_size = label_size
		self.min_std, self.use_lvar = min_std, use_lvar
		if (isnt(rt_params)):
			rt_params = {}

		self.cc_dim = get_xy_concat_dim(rt_name)
		rt_in_shape = get_rt_in_shape(self.in_shape, self.label_size, self.cc_dim)
		self.rep_transform = MODEL_MAPPING.get(rt_name, None)
		try:
			self.rep_transform = self.rep_transform and \
				self.rep_transform(rt_in_shape, **rt_params)
		except Exception as err:
			print("Error! np_util2.py > LatEncoder > __init__()\n",
				sys.exc_info()[0], err)
			print(f'{rt_in_shape=}')
			print(f'{rt_name=}')
			print(f'{rt_params=}')
			raise err

		self.dist_type = dist_type
		self.dist_fn = {
			'beta': torch.distributions.Beta,
			'normal': torch.distributions.Normal,
			'lognormal': torch.distributions.LogNormal
		}.get(self.dist_type, None)
		assert is_valid(self.dist_fn), \
			'dist_type must be a valid latent distribution type'

		self.embed_size = self.rep_transform.out_shape[0] if (is_valid(rt_name)) \
			else rt_in_shape[0]
		self.latent_size = latent_size
		self.map_layer = nn.Linear(self.embed_size, self.embed_size)	# attn -> latent
		self.alpha = nn.Linear(self.embed_size, self.latent_size)	# latent param 1
		self.beta = nn.Linear(self.embed_size, self.latent_size)	# latent param 2

		self.sig = None
		if (self.dist_type.endswith('normal')):
			self.sig = nn.Sigmoid()
			# self.sig = nn.LogSigmoid() if (self.use_lvar) else nn.Sigmoid() XXX
		self.out_shape = (self.latent_size,)

	def forward(self, h, y):
		"""
		ANP Latent Convolutional Encoder forward Pass

		* TODO: aggregate by class and then concatenate instead of global aggregation
		"""
		reps = pt_concat_xy(h, y, self.label_size, dim=self.cc_dim).squeeze()
		enc = self.rep_transform(reps) if (self.rep_transform) else reps

		sa_mean = torch.relu(self.map_layer(reps.mean(dim=2))) # Aggregate over sequence dim
		lat_dist_alpha, lat_dist_beta = self.alpha(sa_mean), self.beta(sa_mean)

		if (self.dist_type.endswith('normal')):
			if (self.use_lvar):
				# Variance clipping in the log domain (should be more stable)
				lat_dist_beta = self.sig(lat_dist_beta)
				lat_dist_beta = torch.clamp(lat_dist_beta, np.log(self.min_std), \
					-np.log(self.min_std))
				lat_dist_sigma = torch.exp(0.5 * lat_dist_beta)
			else:
				# Simple variance clipping (from deep mind repo)
				lat_dist_sigma = self.min_std + (1 - self.min_std) * \
					self.sig(lat_dist_beta)
					# self.sig(lat_dist_beta * 0.5) XXX
			lat_dist_beta = lat_dist_sigma

		lat_dist = self.dist_fn(lat_dist_alpha, lat_dist_beta)
		return lat_dist, lat_dist_beta


class Decoder(nn.Module):
	"""
	ANP Decoder.
		* mean and logsig layers are linear layers
		* decoder output is flattened to be able to send to mean and logsig layers
	"""
	def __init__(self, in_shape, out_size, use_det_path, use_lat_path,
		det_encoder, lat_encoder,
		de_name='ttcn', de_params=None,
		dist_type='beta', min_std=.01, use_lvar=False):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor,
				expects a shape (in_channels, in_height, in_width)
			det_encoder (): deterministic encoder
			lat_encoder (): latent encoder
			de_name (str): decoder name
			de_params (dict): decoder hyperparameters
			dist_type ('beta'|'normal'|'lognormal'|'bernoulli'|'categorical'):
				output distribution
			min_std (float): value used to limit output distribution sigma
			use_lvar (bool): whether to use log domain variance clipping
		"""
		super().__init__()
		self.in_shape = in_shape
		self.out_shape = (out_size,)
		self.min_std, self.use_lvar = min_std, use_lvar
		self.de_name = de_name
		if (isnt(de_params)):
			de_params = {}

		# assert(xt_size == lat_encoder.latent_size)
		# if (use_det_path):
			# assert(xt_size == det_encoder.q_size)

		if (de_name.endswith('tcn')):
			de_chan = 1
			de_chan = de_chan+1 if (use_det_path) else de_chan
			de_chan = de_chan+1 if (use_lat_path) else de_chan
			de_height = self.in_shape[0]
			de_width = self.in_shape[2]
			de_in_shape = (de_chan, de_height, de_width)
			self.decoder = MODEL_MAPPING[de_name](de_in_shape, **de_params)
		elif (de_name == 'ffn'):
			de_in_shape = (self.in_shape[0], self.in_shape[2]) # (height, width) dims
			self.decoder = MODEL_MAPPING[de_name](de_in_shape, **de_params)

		self.dist_type = dist_type
		self.dist_fn = {
			'bernoulli': torch.distributions.Bernoulli,
			'categorical': torch.distributions.Categorical,
			'beta': torch.distributions.Beta,
			'normal': torch.distributions.Normal,
			'lognormal': torch.distributions.LogNormal
		}.get(self.dist_type, None)
		assert is_valid(self.dist_fn), \
			'dist_type must be a valid output distribution type'

		decoder_size = reduce(mul, self.decoder.out_shape)
		self.alpha = nn.Linear(decoder_size, out_size) # primary out_dist parameter
		if (self.dist_type in ('bernoulli', 'categorical')):
			self.beta = None
			self.clamp = nn.Sigmoid()
		elif (self.dist_type in ('beta',)):
			self.beta = nn.Linear(decoder_size, out_size)
			self.clamp = partial(torch.clamp, min=0.0, max=1.0) #nn.Softplus()
		elif (self.dist_type.endswith('normal')):
			self.beta = nn.Linear(decoder_size, out_size)
			self.clamp = partial(torch.clamp, min=0.0, max=1.0) if (self.use_lvar) \
				else nn.Softplus()

	def forward(self, det_rep, lat_rep, target_h):
		"""
		Args:
			det_rep (torch.tensor): deterministic representation (nullable)
			lat_rep (torch.tensor): global latent dist realization
			target_x (torch.tensor): target
		"""
		decoder_inputs = [target_h.squeeze()]
		if (is_valid(lat_rep)):
			lat_rep = lat_rep.unsqueeze(2).expand(-1, -1, target_h.shape[-1]) #XXX
			decoder_inputs.append(lat_rep)
		if (is_valid(det_rep)):
			decoder_inputs.append(det_rep)

		if (self.de_name == 'ttcn'):
			# Multichannel 2d conv -> single channel 2d conv
			# rep = torch.stack(decoder_inputs, dim=2)
			rep = torch.cat(decoder_inputs, dim=1)#.unsqueeze(1)
		elif (self.de_name == 'stcn'):
			# Single channel 2D conv
			rep = torch.cat(decoder_inputs, dim=1)#.unsqueeze(1)
		elif (self.de_name == 'ffn'):
			rep = torch.cat(decoder_inputs, dim=1)
		decoded = self.decoder(rep)

		# For now we're flattening the decoder output embedding
		# out_dist_alpha = self.clamp(self.alpha(torch.flatten(decoded, start_dim=1, end_dim=-1))) # clf
		out_dist_alpha = self.alpha(torch.flatten(decoded, start_dim=1, end_dim=-1)) # reg

		if (self.dist_type in ('bernoulli', 'categorical')):
			out_dist = self.dist_fn(probs=out_dist_alpha)
		elif (self.dist_type in ('beta',)):
			out_dist_beta = self.clamp(self.beta(torch.flatten(decoded, start_dim=1, end_dim=-1)))
			out_dist = self.dist_fn(out_dist_alpha, out_dist_beta)
			# print('a/b')
			# print(out_dist_alpha.squeeze())
			# print(out_dist_beta.squeeze())
		elif (self.dist_type.endswith('normal')):
			out_dist_beta = self.clamp(self.beta(torch.flatten(decoded, start_dim=1, end_dim=-1)))
			if (self.use_lvar):
				out_dist_beta = torch.clamp(out_dist_beta, math.log(self.min_std), \
					-math.log(self.min_std))
				out_dist_sigma = torch.exp(out_dist_beta)
			else:
				out_dist_sigma = self.min_std + (1 - self.min_std) \
					* self.clamp(out_dist_beta)
			out_dist_beta = out_dist_sigma # Bounded or clamped variance
			out_dist = self.dist_fn(out_dist_alpha, out_dist_beta)

		return out_dist 


# ********** MODEL MODULES **********
class AttentiveNP(nn.Module):
	"""
	Attentive Neural Process Module
	"""
	def __init__(self, in_shape, out_size=None, label_size=1, ft_name='stcn', ft_params=None,
		use_det_path=True, use_lat_path=True,
		det_encoder_params=None, lat_encoder_params=None, decoder_params=None,
		sample_latent=True, use_lvar=False, context_in_target=False):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor,
				expects a shape (in_channels, in_height, in_width)
			label_size (int>0): size of the label vector
			det_encoder_params (dict): deterministic encoder hyperparameters
			lat_encoder_params (dict): latent encoder hyperparameters
			decoder_params (dict): decoder hyperparameters
			use_lvar (bool):
			context_in_target (bool):
		"""
		super().__init__()
		self.in_shape = in_shape
		self.label_size = label_size
		self.sample_latent = sample_latent
		self.use_lvar, self.context_in_target = use_lvar, context_in_target
		if (isnt(ft_params)):
			ft_params = {}
		if (isnt(det_encoder_params)):
			det_encoder_params = {}
		if (isnt(lat_encoder_params)):
			lat_encoder_params = {}
		if (isnt(decoder_params)):
			decoder_params = {}

		# Sequence transform (TCN, RNN, etc)
		self.feat_transform = MODEL_MAPPING.get(ft_name, None)
		self.feat_transform = self.feat_transform and \
			self.feat_transform(self.in_shape, **ft_params)
		embed_size = self.feat_transform.out_shape[0]
		enc_in_shape = self.feat_transform.out_shape \
			if (is_valid(self.feat_transform)) else self.in_shape
		dec_in_shape = list(enc_in_shape)

		self.det_encoder, self.lat_encoder = None, None
		if (use_lat_path):
			self.lat_encoder = LatEncoder(enc_in_shape, label_size, **lat_encoder_params)
			dec_in_shape[0] += self.lat_encoder.out_shape[0]
		if (use_det_path):
			self.det_encoder = DetEncoder(enc_in_shape, label_size, embed_size,
				**det_encoder_params)
			dec_in_shape[0] += self.det_encoder.out_shape[0]
		dec_in_shape = tuple(dec_in_shape)

		self.decoder = Decoder(dec_in_shape, out_size or label_size,
			use_det_path, use_lat_path, self.det_encoder, self.lat_encoder,
			**decoder_params)
		self.out_shape = self.decoder.out_shape

	def forward(self, context_x, context_y, target_x, target_y=None):
		"""
		Propagate context and target through neural process network.

		Args:
			context_x (torch.tensor):
			context_y (torch.tensor):
			target_x (torch.tensor):
			target_y (torch.tensor):
			train_mode (bool): whether the model is in training or not.
				If in training, the model will use the (target_x,target_y)
				conditioned posterior as the global latent.

		Returns:
			prior, posterior, and output distribution objects
		"""
		if (self.feat_transform):
			context_h = self.feat_transform(context_x)
			target_h = self.feat_transform(target_x)
		else:
			context_h, target_h = context_x, target_x
		det_rep, lat_rep, prior_dist, post_dist = None, None, None, None

		if (is_valid(self.det_encoder)):
			det_rep = self.det_encoder(context_h, context_y, target_h)

		if (is_valid(self.lat_encoder)):
			prior_dist, prior_beta = self.lat_encoder(context_h, context_y)

			if (is_valid(target_y)):
				# At training time:
				post_dist, post_beta = self.lat_encoder(target_h, target_y)
				lat_rep = post_dist.rsample() if (self.sample_latent) else post_dist.mean
			else:
				# At test/inference time:
				post_dist, post_beta = None, None
				lat_rep = prior_dist.rsample() if (self.sample_latent) else prior_dist.mean

		out_dist = self.decoder(det_rep, lat_rep, target_h)
		return prior_dist, post_dist, out_dist

	@classmethod
	def suggest_params(cls, trial=None, num_classes=2):
		"""
		suggest model hyperparameters from an optuna trial object
		or return fixed default hyperparameters
		"""
		params_xa_mha = {
			'heads': 1,
			'dropout': 0.0,
			'depth': 1,
		}

		params_ffn = {
			'out_shapes': [32, 32, 32],
			'act': 'relu',
			'init': 'xavier_uniform',
		}

		params_stcn = {
			'size': 6,
			'depth': 4,
			'kernel_sizes': 15,
			'input_dropout': 0.0,
			'output_dropout': 0.0,
			'global_dropout': 0.0,
			'global_dilation': True,
			'block_act': 'relu',
			'out_act': 'relu',
			'block_init': 'kaiming_uniform',
			'out_init': 'kaiming_uniform',
			'pad_mode': 'full'
		}

		params_ttcn = {
			'channels': [3, 1],
			'kernel_sizes': 15,
			'input_dropout': 0.0,
			'output_dropout': 0.0,
			'global_dropout': 0.0,
			'use_dilation': True,
			'block_act': 'relu',
			'out_act': 'relu',
			'block_init': 'kaiming_uniform',
			'out_init': 'kaiming_uniform',
			'pad_mode': 'full',
			'tdims': (2, 1),
			'use_residual': True,
			'downsample_type': 'conv2d'
		}

		if (is_valid(trial)):
			pass
		else:
			params = {
				'ft_name': 'stcn', 'ft_params': params_stcn,
				'det_encoder_params': {
					'rt_name': 'ffn', 'rt_params': params_ffn,
					'xa_name': 'mha', 'xa_params': params_xa_mha,
				},
				'lat_encoder_params': {
					'latent_size': 256,
					'rt_name': 'ffn', 'rt_params': params_ffn,
					'dist_type': 'normal', 'min_std': .01, 'use_lvar': False
				},
				'decoder_params': {
					'de_name': 'ttcn', 'de_params': params_ttcn,
					'dist_type': 'normal', 'min_std': .01, 'use_lvar': False
				},
				'use_det_path': True,
				'use_lat_path': True,
				'sample_latent': True,
				'use_lvar': False,
				'context_in_target': False,
				'label_size': num_classes-1
			}

		return params

