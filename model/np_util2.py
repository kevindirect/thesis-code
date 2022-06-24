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
from model.model_util import log_prob_sigma, init_layer, get_padding, pt_multihead_attention, SwapLinear, TransposeModule, MODEL_MAPPING, NORM_MAPPING
# Tensors are column-major, shaped as (batch, channel, height, width) unless otherwise specified
# The in_shape and out_shape attributes of modules don't include the batch size
# Inspired by:
# * https://github.com/deepmind/neural-processes/blob/master/attentive_neural_process.ipynb
# * https://github.com/3springs/attentive-neural-processes/blob/master/neural_processes/models/neural_process/model.py
# * https://github.com/soobinseo/Attentive-Neural-Process/blob/master/module.py


# ********** HELPER FUNCTIONS **********
def pt_cat_xy(x, y, size=1, dim=2):
	assert(x.shape[0] == y.shape[0])
	xpd_y = y.reshape(y.shape[0], 1, 1, 1).float()

	if (dim == 1):
		xpd_y = xpd_y.expand(-1, size, *x.shape[2:])
	elif (dim == 2):
		xpd_y = xpd_y.expand(-1, x.shape[1], size, x.shape[3])
	elif (dim == 3):
		xpd_y = xpd_y.expand(-1, *x.shape[1:3], size)

	return torch.cat([x, xpd_y], dim=dim)

def get_xy_cat_dim(rt_name):
	"""
	Return dimension to cat x and y tensors (includes batch size)
	"""
	return {
		'stcn': 2,
		'ffn': 3
	}.get(rt_name, 3)

def get_rt_in_shape(in_shape, label_size, cat_dim):
	"""
	Note: input and output shape variables do not start with a batch size
	"""
	rt_in_shape = list(in_shape)
	rt_in_shape[cat_dim-1] += label_size # cat_dim-1 to account for no batch size
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
		self.class_agg = class_agg

		# padding to make len(key_seq)==len(value_seq) after value label append:
		self.key_padding = nn.ConstantPad1d((0, self.label_size), 0.0)

		if (isnt(rt_params)):
			rt_params = {}
		if (isnt(xa_params)):
			xa_params = {}

		self.cat_dim = get_xy_cat_dim(rt_name)
		rt_in_shape = get_rt_in_shape(self.in_shape, self.label_size, self.cat_dim)
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

		if (self.class_agg):
			self.cross_aggregation = nn.ModuleList()
			xa_in_shape = self.rep_transform.out_shape if (is_valid(rt_name)) \
				else rt_in_shape
			kdim, vdim = self.in_shape[0], xa_in_shape[0]
			cross_agg_fn = MODEL_MAPPING[xa_name]
			self.class_agg_method = 'cat' # 'add' or 'cat'

			for i in range(num_classes := self.label_size+1):
				try:
					self.cross_aggregation.append(cross_agg_fn(xa_in_shape,
						kdim=kdim, vdim=vdim, **xa_params))
				except Exception as err:
					print("Error! np_util2.py > DetEncoder > __init__ > cross_agg_fn()\n",
						sys.exc_info()[0], err)
					print(f'{i=}')
					print(f'{num_classes=}')
					print(f'{xa_in_shape=}')
					print(f'{xa_name=}')
					print(f'{xa_params=}')
					raise err
			self.out_shape = self.cross_aggregation[0].out_shape
		else:
			xa_in_shape = self.rep_transform.out_shape if (is_valid(rt_name)) \
				else rt_in_shape
			kdim, vdim = self.in_shape[0], xa_in_shape[0]
			cross_agg_fn = MODEL_MAPPING[xa_name]
			try:
				self.cross_aggregation = cross_agg_fn(xa_in_shape,
					kdim=kdim, vdim=vdim, **xa_params)
			except Exception as err:
				print("Error! np_util2.py > DetEncoder > __init__ > cross_agg_fn()\n",
					sys.exc_info()[0], err)
				print(f'{xa_in_shape=}')
				print(f'{xa_name=}')
				print(f'{xa_params=}')
				print(f'{kdim=}')
				print(f'{vdim=}')
				raise err
			self.out_shape = self.cross_aggregation.out_shape

	def forward(self, context_h, context_y, target_h):
		"""
		ANP Deterministic Encoder forward Pass
		Returns all representations, need to be aggregated to form r_*.
		* Include Positional Encoding?
		"""
		queries, keys = target_h.squeeze(), self.key_padding(context_h.squeeze())
		reps = pt_cat_xy(context_h, context_y, self.label_size, dim=self.cat_dim) \
			.squeeze()
		values = self.rep_transform(reps) if (self.rep_transform) else reps

		# have cross aggregation ignore key padding that we appended to keys:
		key_padding_mask = keys.new_zeros((keys.shape[0], keys.shape[-1]), \
			dtype=torch.bool)
		key_padding_mask[:, -1] = True # True -> ignored

		if (self.class_agg):
			classes = context_y.unique(sorted=True, return_inverse=False)
			cla_reps = []
			for i, cla in enumerate(classes):
				# dont attend to examples of other classes:
				attn_mask = values.new_zeros((values.shape[0], queries.shape[-1], values.shape[-1]), \
					dtype=torch.bool)
				attn_mask[context_y != cla, :, :] = True # True -> ignored
				num_heads = self.cross_aggregation[i].num_heads
				attn_mask = attn_mask.repeat_interleave(num_heads, dim=0) # repeat each batch example for each head
				# attn_mask = attn_mask.repeat(num_heads, 1, 1)

				cla_rep = self.cross_aggregation[i](queries, keys, values, \
					key_padding_mask=key_padding_mask, attn_mask=attn_mask)

				if (self.class_agg_method == 'add'):
					cla_rep[torch.isnan(cla_rep)] = 0.0
				elif (self.class_agg_method == 'cat'):
					cla_rep = cla_rep[context_y == cla]
				cla_reps.append(cla_rep)

			if (self.class_agg_method == 'add'):
				det_rep = reduce(torch.add, cla_reps)
			elif (self.class_agg_method == 'cat'):
				det_rep = torch.cat(cla_reps, dim=0)
		else:
			det_rep = self.cross_aggregation(queries, keys, values, \
				key_padding_mask=key_padding_mask, attn_mask=None)

		return det_rep


class LatEncoder(nn.Module):
	"""
	Attentive Neural Process (ANP) Latent Encoder.

	1. Transform (Xc, yc) -> enc
	2. Run self attention on enc
	3. Use aggregate representation of enc to parameterize a distribution (latent variable)
	* using linear layers for map_layer, alpha, and beta modules
	"""
	def __init__(self, in_shape, label_size, latent_size=256, cat_before_rt=True, class_agg=False,
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
		self.cat_before_rt = cat_before_rt
		self.class_agg = class_agg
		if (isnt(rt_params)):
			rt_params = {}

		self.cat_dim = get_xy_cat_dim(rt_name)
		rt_in_shape = get_rt_in_shape(self.in_shape, self.label_size, self.cat_dim)
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

		self.out_chan = self.rep_transform.out_shape[0] if (is_valid(rt_name)) \
			else rt_in_shape[0]
		self.out_chan = (self.label_size+1) * self.out_chan if (self.class_agg) \
			else self.out_chan
		self.embed_size = self.rep_transform.out_shape[-1] if (is_valid(rt_name)) \
			else rt_in_shape[-1]
		self.latent_size = latent_size if (is_valid(latent_size)) \
			else self.in_shape[-2] * self.in_shape[-1]
		self.map_size = (self.embed_size + self.latent_size) // 2
		self.map_layer = nn.Linear(self.embed_size, self.map_size)
		self.alpha = nn.Linear(self.map_size, self.latent_size)	# latent param 1
		self.beta = nn.Linear(self.map_size, self.latent_size)	# latent param 2

		self.sig = None
		if (self.dist_type.endswith('normal')):
			self.sig = nn.Sigmoid()
			# self.sig = nn.LogSigmoid() if (self.use_lvar) else nn.Sigmoid() XXX
		elif (self.dist_type in ('beta',)):
			self.sig = nn.Softplus()
		self.out_shape = (self.out_chan, self.latent_size)

	def forward(self, h, y):
		"""
		ANP Latent Convolutional Encoder forward Pass

		h is shaped (n, channels, height, width)
		y is shaped (n, )
		"""
		if (self.cat_before_rt):
			enc = pt_cat_xy(h, y, self.label_size, dim=self.cat_dim).squeeze()
			enc = self.rep_transform(enc) if (self.rep_transform) else enc
		else:
			enc = self.rep_transform(h.squeeze()).unsqueeze(self.cat_dim-1) \
				if (self.rep_transform) else h
			enc = pt_cat_xy(enc, y, self.label_size, dim=self.cat_dim).squeeze()

		# mean representation of encoded batch:
		if (self.class_agg):
			classes = y.unique(sorted=True, return_inverse=False)
			cla_enc_means = []
			for i, cla in enumerate(classes):
				cla_enc = enc[y == cla]
				cla_enc_mean = cla_enc.mean(dim=0)
				cla_enc_means.append(cla_enc_mean)
			enc_mean = torch.cat(cla_enc_means, dim=0)
			# print(f'{enc_mean.shape=}')
			# print(f'{enc_mean=}')
			enc_param = torch.relu(self.map_layer(enc_mean))
			# print(f'{enc_param.shape=}')
			# print(f'{enc_param=}')
			lat_dist_alpha, lat_dist_beta = self.alpha(enc_param), self.beta(enc_param)
		else:
			enc_mean = enc.mean(dim=0)
			enc_param = torch.relu(self.map_layer(enc_mean))
			lat_dist_alpha, lat_dist_beta = self.alpha(enc_param), self.beta(enc_param)

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
		elif (self.dist_type in ('beta',)):
			lat_dist_alpha = self.sig(lat_dist_alpha)
			lat_dist_beta = self.sig(lat_dist_beta)

		lat_dist = self.dist_fn(lat_dist_alpha, lat_dist_beta)
		return lat_dist, lat_dist_beta


class Decoder(nn.Module):
	"""
	ANP Decoder.
		* mean and logsig layers are linear layers
		* decoder output is flattened to be able to send to mean and logsig layers
	"""
	def __init__(self, in_shape, out_size, use_raw, use_det_path, use_lat_path,
		det_encoder, lat_encoder, de_name='ffn', de_params=None,
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
		self.out_size = out_size
		self.min_std, self.use_lvar = min_std, use_lvar
		self.de_name = de_name
		self.use_raw = use_raw
		if (isnt(de_params)):
			de_params = {}

		# assert(xt_size == lat_encoder.latent_size)
		# if (use_det_path):
			# assert(xt_size == det_encoder.q_size)

		if (de_name.endswith('tcn')):
			de_chan = 0
			de_chan = de_chan+1 if (self.use_raw) else de_chan
			de_chan = de_chan+1 if (use_det_path) else de_chan
			de_chan = de_chan+1 if (use_lat_path) else de_chan
			de_height = self.in_shape[0]
			de_width = self.in_shape[-1]
			de_in_shape = (de_chan, de_height, de_width)
			self.decoder = MODEL_MAPPING[de_name](de_in_shape, **de_params)
		elif (de_name == 'ffn'):
			de_chan = self.in_shape[0]
			de_width = self.in_shape[-1]
			if (de_params['flatten']):
				de_params['out_shapes'].append(self.out_size)
				self.decoder = MODEL_MAPPING[de_name]((de_chan, de_width), **de_params)
			else:
				de_seq_params = {**de_params}
				de_chan_params = {**de_params}
				de_seq_out_shapes = [de_width]*len(de_params['out_shapes']) + [1]
				de_chan_out_shapes = [de_chan]*len(de_params['out_shapes']) + [self.out_size]
				# de_seq_params['out_shapes'].append(1)
				de_seq_params['out_shapes'] = de_seq_out_shapes
				de_chan_params['out_shapes'] = de_chan_out_shapes
				de_seq = MODEL_MAPPING[de_name]((de_chan, de_width), **de_seq_params)
				de_chan = MODEL_MAPPING[de_name]((1, de_chan), **de_chan_params)
				tm1 = TransposeModule(1, -1)
				self.decoder = nn.Sequential(de_seq, tm1, de_chan)

		if (self.de_name == 'ttcn'):
			# Multichannel 2d conv -> single channel 2d conv
			# rep = torch.stack(decoder_inputs, dim=2)
			# rep = torch.cat(decoder_inputs, dim=1)#.unsqueeze(1)
			self.cat_dim = 1
		elif (self.de_name == 'stcn'):
			# Single channel 2D conv
			# rep = torch.cat(decoder_inputs, dim=1)#.unsqueeze(1)
			self.cat_dim = 1
		elif (self.de_name == 'ffn'):
			self.cat_dim = 1 # XXX cat over C(1) or S(2)

		self.dist_type = dist_type
		self.dist_fn = {
			'bernoulli': torch.distributions.Bernoulli,
			'categorical': torch.distributions.Categorical,
			'beta': torch.distributions.Beta,
			'normal': torch.distributions.Normal,
			'mvnormal': torch.distributions.MultivariateNormal,
			'lognormal': torch.distributions.LogNormal
		}.get(self.dist_type, None)
		assert is_valid(self.dist_fn), \
			'dist_type must be a valid output distribution type'

		# dec_size = reduce(mul, self.decoder.out_shape)
		# dec_chan, dec_size = self.decoder.out_shape[0], self.decoder.out_shape[-1]
		self.alpha = nn.Linear(self.out_size, self.out_size) # primary out_dist parameter
		if (self.dist_type in ('bernoulli', 'categorical')):
			self.beta = None
			self.clamp = nn.Sigmoid()
			self.out_shape = (self.out_size,)
		elif (self.dist_type in ('beta',)):
			self.beta = nn.Linear(self.out_size, self.out_size)
			self.clamp = nn.Softplus(beta=1, threshold=20)
			self.out_shape = (self.out_size,)
		elif (self.dist_type in ('normal', 'lognormal')):
			self.beta = nn.Linear(self.out_size, self.out_size)
			self.clamp = partial(torch.clamp, min=0.0, max=1.0) if (self.use_lvar) \
				else nn.Softplus()
			self.out_shape = (self.out_size,)
		elif (self.dist_type in ('mvnormal',)):
			self.beta = nn.Linear(self.out_size, self.out_size**2)
			self.clamp = partial(torch.clamp, min=0.0, max=1.0) if (self.use_lvar) \
				else nn.Softplus(beta=1, threshold=20)
			self.out_shape = (self.out_size,)

	def forward(self, det_rep, lat_rep, target_h):
		"""
		Args:
			det_rep (torch.tensor): deterministic representation (nullable)
			lat_rep (torch.tensor): global latent dist realization
			target_x (torch.tensor): target
		"""
		decoder_inputs = []
		if (self.use_raw):
			decoder_inputs.append(target_h.squeeze())
		if (is_valid(det_rep)):
			decoder_inputs.append(det_rep)
		if (is_valid(lat_rep)):
			noop = [1] * lat_rep.ndim
			lat_rep = lat_rep.unsqueeze(0).repeat(target_h.shape[0], *noop)
			decoder_inputs.append(lat_rep)

		try:
			rep = torch.cat(decoder_inputs, dim=self.cat_dim)
		except Exception as err:
			print("Error! np_util2.py > Decoder > forward() > torch.cat()\n",
				sys.exc_info()[0], err)
			print(f'{self.cat_dim=}')
			for i in range(len(decoder_inputs)):
				print(f'{decoder_inputs[i].shape=}')

		decoded = self.decoder(rep)
		out_dist_alpha = self.alpha(decoded)

		if (self.dist_type in ('bernoulli', 'categorical')):
			out_dist_alpha = self.clamp(out_dist_alpha.squeeze())
			out_dist = self.dist_fn(probs=out_dist_alpha)
		elif (self.dist_type in ('beta',)):
			out_dist_beta = self.beta(decoded)
			out_dist_alpha = self.clamp(out_dist_alpha.squeeze())
			out_dist_beta = self.clamp(out_dist_beta.squeeze())
			out_dist = self.dist_fn(out_dist_alpha, out_dist_beta)
		elif (self.dist_type in ('normal', 'lognormal')):
			out_dist_beta = self.beta(decoded)
			if (self.use_lvar):
				out_dist_beta = torch.clamp(out_dist_beta, math.log(self.min_std), \
					-math.log(self.min_std))
				out_dist_sigma = torch.exp(out_dist_beta)
			else:
				out_dist_sigma = self.min_std + (1 - self.min_std) \
					* self.clamp(out_dist_beta)
			out_dist_beta = out_dist_sigma # Bounded or clamped variance
			out_dist = self.dist_fn(out_dist_alpha.squeeze(), out_dist_beta.squeeze())
		elif (self.dist_type in ('mvnormal',)):
			out_dist_beta = self.beta(decoded) \
				.reshape(decoded.shape[0], -1, self.out_size, self.out_size)
			if (self.use_lvar):
				out_dist_beta = torch.clamp(out_dist_beta, math.log(self.min_std), \
					-math.log(self.min_std))
				out_dist_sigma = torch.exp(out_dist_beta)
			else:
				out_dist_sigma = self.min_std + (1 - self.min_std) \
					* self.clamp(out_dist_beta)
			# lower triangle of covariance matrix:
			out_dist_beta = torch.tril(out_dist_sigma)
			out_dist = self.dist_fn(out_dist_alpha, scale_tril=out_dist_beta)

		return out_dist


# ********** MODEL MODULES **********
class AttentiveNP(nn.Module):
	"""
	Attentive Neural Process Module
	"""
	def __init__(self, in_shape, out_size=None, label_size=1,
		in_name='in2d', in_params=None, in_split=False,
		fn_name=None, fn_params=None, fn_split=False,
		ft_name='stcn', ft_params=None, use_raw=True, use_det_path=True, use_lat_path=True,
		det_encoder_params=None, lat_encoder_params=None, decoder_params=None,
		sample_latent_post=True, sample_latent_prior=False, use_lvar=False):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor,
				expects a shape (in_channels, in_height, in_width)
			label_size (int>0): size of the label vector
			det_encoder_params (dict): deterministic encoder hyperparameters
			lat_encoder_params (dict): latent encoder hyperparameters
			decoder_params (dict): decoder hyperparameters
			use_lvar (bool):
		"""
		super().__init__()
		self.in_shape = in_shape
		self.label_size = label_size
		self.sample_latent_post = sample_latent_post
		self.sample_latent_prior = sample_latent_prior
		self.use_lvar = use_lvar
		if (isnt(ft_params)):
			ft_params = {}
		if (isnt(det_encoder_params)):
			det_encoder_params = {}
		if (isnt(lat_encoder_params)):
			lat_encoder_params = {}
		if (isnt(decoder_params)):
			decoder_params = {}

		self.input_norm_fn = NORM_MAPPING.get(in_name, None)
		if (is_valid(self.input_norm_fn)):
			self.context_input_norm = self.input_norm_fn(self.in_shape, **in_params)
			if (in_split):
				self.target_input_norm = self.input_norm_fn(self.in_shape, **in_params)
			else:
				self.target_input_norm = self.context_input_norm

		# Sequence transform (TCN, RNN, etc)
		self.feat_transform_fn = MODEL_MAPPING.get(ft_name, None)
		if (is_valid(self.feat_transform_fn)):
			self.feat_transform = self.feat_transform_fn(self.in_shape, **ft_params)
			enc_in_shape = self.feat_transform.out_shape
		else:
			enc_in_shape = self.in_shape

		embed_size = enc_in_shape[0]
		dec_in_shape = list(enc_in_shape)
		if (not use_raw):
			dec_in_shape[0] = 0

		self.feat_norm_fn = NORM_MAPPING.get(fn_name, None) # normalizations don't change tensor shape, so it's safe
		if (self.feat_norm_fn):
			self.context_feat_norm = self.feat_norm_fn(enc_in_shape, **fn_params)
			if (fn_split):
				self.target_feat_norm = self.feat_norm_fn(enc_in_shape, **fn_params)
			else:
				self.target_feat_norm = self.context_feat_norm

		self.det_encoder, self.lat_encoder = None, None
		if (use_lat_path):
			self.lat_encoder = LatEncoder(enc_in_shape, label_size,
				**lat_encoder_params)
			if (self.lat_encoder.out_shape[-1] != dec_in_shape[-1]):
				self.lat_downsample = nn.Linear(self.lat_encoder.out_shape[-1], dec_in_shape[-1])
			else:
				self.lat_downsample = None
			dec_in_shape[0] += self.lat_encoder.out_shape[0]
			# print(f'{self.lat_encoder.in_shape=}')
			# print(f'{self.lat_encoder.out_shape=}')
		if (use_det_path):
			self.det_encoder = DetEncoder(enc_in_shape, label_size, embed_size,
				**det_encoder_params)
			dec_in_shape[0] += self.det_encoder.out_shape[0]
			# print(f'{self.det_encoder.in_shape=}')
			# print(f'{self.det_encoder.out_shape=}')
		dec_in_shape = tuple(dec_in_shape)
		assert dec_in_shape[0] > 0, 'decoder must take in data; decoder in channels must be greater than 0'

		self.decoder = Decoder(dec_in_shape, out_size or label_size, use_raw,
			use_det_path, use_lat_path, self.det_encoder, self.lat_encoder,
			**decoder_params)
		# print(f'{self.decoder.in_shape=}')
		# print(f'{self.decoder.out_shape=}')
		self.out_shape = self.decoder.out_shape

	def forward(self, context_x, context_y, target_x, target_y=None):
		"""
		Propagate context and target through neural process network.

		Args:
			context_x (torch.tensor):
			context_y (torch.tensor):
			target_x (torch.tensor):
			target_y (torch.tensor):

		Returns:
			prior, posterior, and output distribution objects
		"""
		if (is_valid(self.input_norm_fn)):
			context_x = self.context_input_norm(context_x)
			target_x = self.target_input_norm(target_x)

		context_h, target_h = context_x, target_x

		if (is_valid(self.feat_transform_fn)):
			context_h = self.feat_transform(context_h)
			target_h = self.feat_transform(target_h)

		if (is_valid(self.feat_norm_fn)):
			context_h = self.context_feat_norm(context_h)
			target_h = self.target_feat_norm(target_h)

		det_rep = lat_rep = prior_dist = post_dist = None

		if (is_valid(self.det_encoder)):
			det_rep = self.det_encoder(context_h, context_y, target_h)

		if (is_valid(self.lat_encoder)):
			prior_dist, prior_beta = self.lat_encoder(context_h, context_y)

			if (is_valid(target_y)):
				# At training time:
				post_dist, post_beta = self.lat_encoder(target_h, target_y)
				lat_rep = post_dist.rsample() if (self.sample_latent_post) \
					else post_dist.mean
			else:
				# At eval/test time:
				post_dist, post_beta = None, None
				lat_rep = prior_dist.rsample() if (self.sample_latent_prior) \
					else prior_dist.mean

			if (is_valid(self.lat_downsample)):
				lat_rep = self.lat_downsample(lat_rep)

		out_dist = self.decoder(det_rep, lat_rep, target_h)
		return prior_dist, post_dist, out_dist

