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
from model.common import PYTORCH_ACT_MAPPING
from model.model_util import log_prob_sigma, init_layer, get_padding, pt_multihead_attention, SwapLinear, TransposeModule, MODEL_MAPPING, NORM_MAPPING


# ********** HELPER FUNCTIONS **********
def get_xy_shape(self, obs_size, x_shape, y_size):
	xy_shape = [obs_size, *x_shape]
	xy_shape[-1] += y_size
	return tuple(xy_shape)

def collapse_lower(x, start_dim=0, end_dim=2):
	"""
	Like torch.flatten but always returns a view
	"""
	return x.view(np.product(x.shape[start_dim:end_dim]), *x.shape[end_dim:])

def uncollapse_lower(x, lower, end_dim=1):
	"""
	Uncollapse the lower dimensions back in
	"""
	return x.view(*lower, *x.shape[end_dim:])

def MultivariateNormalDiag(loc, scale_diag):
	"""
	From: https://github.com/pytorch/pytorch/pull/11178
	"""
	if (loc.dim() < 1):
		raise ValueError("loc must be at least one-dimensional.")
	return torch.distributions.Independent(torch.distributions.Normal(loc, scale_diag), 1)

# ********** HELPER MODULES **********
class DetEncoder(nn.Module):
	"""
	Attentive Neural Process (ANP) Deterministic Encoder.
	Returns a representation of the context pairs attended to by targets.

	1. cat(Xc, yc) -> V
	2. Run rep transform (self-attention) on V
	3. Run cross aggregation on Q, K, V to get representation of V (context pairs) based on Q (target features) <-> K (context features) similarity
	"""
	def __init__(self, in_shape, context_size, target_size, out_size,
		rt_name='mha', rt_params=None, xa_name='mha', xa_params=None):
		"""
		Args:
			in_shape (tuple): shape of feature observation
			context_size (int>0): context set size
			target_size (int>0): target set size
			out_size (int>0): size of the label vector
			rt_name (str): representation transform name
			rt_params (dict): representation transform hyperparameters
			xa_name (str): cross aggregation name
			xa_params (dict): cross aggregation hyperparameters
		"""
		super().__init__()
		self.in_shape = in_shape
		self.out_size = out_size

		# padding for value concat
		# self.padding = nn.ConstantPad1d((0, self.out_size), 0.0)

		rt_params = rt_params or {}
		xa_params = xa_params or {}

		self.q_shape = (target_size, *self.in_shape)
		# self.k_shape = (context_size, *self.in_shape)
		self.v_shape = [context_size, *self.in_shape]
		self.v_shape[-1] += self.out_size
		if (is_valid(rep_fn := MODEL_MAPPING.get(rt_name, None))):
			self.rep_transform = rep_fn(self.v_shape, **rt_params)
			self.v_shape = self.rep_transform.out_shape
		else:
			self.rep_transform = None

		xa_fn = MODEL_MAPPING[xa_name]
		self.cross_aggregation = xa_fn(self.q_shape, vdim=self.v_shape[-1], **xa_params)
		self.out_shape = self.cross_aggregation.out_shape

	def forward(self, context_h, context_y, target_h):
		"""
		ANP Deterministic Encoder forward Pass
		"""
		queries = target_h.squeeze()
		keys = context_h.squeeze()
		values = torch.cat((context_h, context_y), dim=-1)

		if (self.rep_transform):
			values = self.rep_transform(values)

		return self.cross_aggregation(queries, keys, values)

class LatEncoder(nn.Module):
	"""
	Attentive Neural Process (ANP) Latent Encoder.

	1. cat(Xc, yc) -> V
	2. Run rep transform (self-attention) on V
	3. Use aggregate representation of enc to parameterize a distribution (latent variable)
	"""
	def __init__(self, in_shape, out_size, latent_size=256,
		rt_name='mha', rt_params=None, dist_type='normal',
		min_std=.01, use_lvar=False):
		"""
		Args:
			in_shape (tuple): shape of feature observation
			out_size (int>0): size of the label vector
			latent_size (int>0): size of latent representation
			rt_name (str): representation transform name
			rt_params (dict): representation transform hyperparameters
			dist_type ('beta'|'normal'|'lognormal'): latent distribution
			min_std (float): value used to limit latent distribution sigma
			use_lvar (bool): whether to use log domain variance clipping
		"""
		super().__init__()
		self.in_shape = in_shape
		self.out_size = out_size
		self.min_std = min_std
		self.use_lvar = use_lvar
		rt_params = rt_params or {}

		self.v_shape = [None, *self.in_shape]
		self.v_shape[-1] += self.out_size
		if (is_valid(rep_fn := MODEL_MAPPING.get(rt_name, None))):
			self.rep_transform = rep_fn(self.v_shape, **rt_params)
			self.v_shape = self.rep_transform.out_shape
		else:
			self.rep_transform = None

		self.dist_type = dist_type
		self.dist_fn = {
			'normal': torch.distributions.Normal
			# 'lognormal': torch.distributions.LogNormal,
			# 'beta': torch.distributions.Beta,
		}.get(self.dist_type, None)
		assert is_valid(self.dist_fn)

		embed_size = self.v_shape[-1]
		latent_size = latent_size or embed_size
		interim_size = (embed_size + latent_size) // 2

		self.interim_layer = init_layer(nn.Linear(embed_size, interim_size))
		self.alpha = init_layer(nn.Linear(interim_size, latent_size))
		self.beta = init_layer(nn.Linear(interim_size, latent_size))

		self.beta_act = None
		if (self.dist_type.endswith('normal')):
			self.beta_act = nn.LogSigmoid() if (self.use_lvar) else nn.Sigmoid()
		# elif (self.dist_type in ('beta',)):
		# 	self.beta_act = nn.Softplus()
		self.out_shape = (latent_size,)

	def forward(self, h, y):
		"""
		ANP Latent Convolutional Encoder forward Pass
		"""
		values = torch.cat((h, y), dim=-1)

		if (self.rep_transform):
			values = self.rep_transform(values)

		enc_mean = values.mean(dim=1) # average over {context, target} set
		enc_param = torch.relu(self.interim_layer(enc_mean))
		lat_dist_alpha, lat_dist_beta = self.alpha(enc_param), self.beta(enc_param)

		if (self.dist_type.endswith('normal')):
			if (self.use_lvar):
				# Variance clipping in the log domain (should be more stable)
				lstd = math.log(self.min_std)
				lat_dist_beta = self.beta_act(lat_dist_beta)
				lat_dist_beta = torch.clamp(lat_dist_beta, lstd, -lstd)
				lat_dist_sigma = torch.exp(0.5 * lat_dist_beta)
			else:
				# Simple variance clipping (from deep mind repo)
				lat_dist_sigma = self.min_std + (1 - self.min_std) * self.beta_act(lat_dist_beta)
			lat_dist_beta = lat_dist_sigma
		# elif (self.dist_type in ('beta',)):
		# 	lat_dist_alpha = self.beta_act(lat_dist_alpha)
		# 	lat_dist_beta = self.beta_act(lat_dist_beta)

		return self.dist_fn(lat_dist_alpha, lat_dist_beta)


class Decoder(nn.Module):
	"""
	ANP Decoder.
		* mean and logsig layers are linear layers
		* decoder output is flattened to be able to send to mean and logsig layers
	"""
	def __init__(self, in_shape, target_size, out_size, use_raw, use_det_path, use_lat_path,
		det_encoder, lat_encoder, de_name='ffn', de_params=None,
		act=None, dist_type='mvnormaldiag', min_std=.01, use_lvar=False):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor (o, W)
			det_encoder (): deterministic encoder
			lat_encoder (): latent encoder
			de_name (str): decoder name
			de_params (dict): decoder hyperparameters
			dist_type: output distribution
			min_std (float): value used to limit output distribution sigma
			use_lvar (bool): whether to use log domain variance clipping
		"""
		super().__init__()
		self.in_shape = in_shape
		self.target_size = target_size
		self.out_size = out_size
		self.min_std, self.use_lvar = min_std, use_lvar
		self.de_name = de_name
		self.use_raw = use_raw
		de_params = de_params or {}

		self.decoder = MODEL_MAPPING[self.de_name]((in_shape[-1],), **de_params)
		assert len(self.decoder.out_shape)==1

		self.dist_type = dist_type
		self.dist_fn = {
			'mvnormal': torch.distributions.MultivariateNormal,
			'mvnormaldiag': MultivariateNormalDiag,
			# 'beta': torch.distributions.Beta,
			# 'normal': torch.distributions.Normal,
			# 'lognormal': torch.distributions.LogNormal
		}.get(self.dist_type, None)
		assert is_valid(self.dist_fn)

		dec_out = self.decoder.out_shape[0]
		self.alpha = init_layer(nn.Linear(dec_out, self.out_size))
		self.alpha_act = (af := PYTORCH_ACT_MAPPING.get(act, None)) and af()
		if (self.dist_type in ('mvnormal', 'mvnormaldiag')):
			self.beta = init_layer(nn.Linear(dec_out, self.out_size))
			self.beta_act = self.use_lvar or nn.Softplus(beta=1, threshold=20)
			self.out_shape = (self.out_size,)
		# elif (self.dist_type in ('beta',)):
		# 	self.beta = nn.Linear(self.out_size, self.out_size)
		# 	self.clamp = nn.Softplus(beta=1, threshold=20)
		# 	self.out_shape = (self.out_size,)
		# elif (self.dist_type in ('normal', 'lognormal')):
		# 	self.beta = nn.Linear(self.out_size, self.out_size)
		# 	self.clamp = partial(torch.clamp, min=0.0, max=1.0) if (self.use_lvar) \
		# 		else nn.Softplus()
		# 	self.out_shape = (self.out_size,)

	def forward(self, det_rep, lat_rep, target_h):
		"""
		Args:
			det_rep (torch.tensor): deterministic representation (nullable)
			lat_rep (torch.tensor): global latent dist realization
			target_h (torch.tensor):
		"""
		decoder_inputs = []
		if (self.use_raw):
			decoder_inputs.append(target_h)
		if (is_valid(det_rep)):
			decoder_inputs.append(det_rep)
		if (is_valid(lat_rep)):
			tiled = lat_rep.unsqueeze(1).expand(-1, self.target_size, -1)
			decoder_inputs.append(tiled)

		rep = torch.cat(decoder_inputs, dim=-1)
		decoded = self.decoder(collapse_lower(rep))
		decoded = uncollapse_lower(decoded, rep.shape[:2])
		out_dist_alpha = self.alpha(decoded)
		if (is_valid(self.alpha_act)):
			out_dist_alpha = self.alpha_act(out_dist_alpha)

		if (self.dist_type in ('mvnormal', 'mvnormaldiag',)):
			out_dist_beta = self.beta(decoded) \
				.reshape(decoded.shape[0], -1, self.out_size, self.out_size)
			if (self.use_lvar):
				lstd = math.log(self.min_std)
				out_dist_beta = torch.clamp(out_dist_beta, lstd, -lstd)
				out_dist_sigma = torch.exp(out_dist_beta)
			else:
				out_dist_sigma = self.min_std + (1 - self.min_std) * self.beta_act(out_dist_beta)
			if (self.dist_type == 'mvnormal'):
				raise NotImplementedError()
				# lower triangle of covariance matrix:
				out_dist_tril = torch.tril(out_dist_sigma)
				out_dist = self.dist_fn(out_dist_alpha, scale_tril=out_dist_tril)
			elif (self.dist_type == 'mvnormaldiag'):
				out_dist = self.dist_fn(out_dist_alpha.squeeze(), out_dist_sigma.squeeze())

		return out_dist


# ********** MODEL MODULES **********
class AttentiveNP(nn.Module):
	"""
	Attentive Neural Process Module
	"""
	def __init__(self, in_shape, context_size, target_size, out_size=1,
		in_name='in15d', in_params=None, in_split=False,
		fn_name=None, fn_params=None, fn_split=False,
		ft_name='stcn', ft_params=None, use_raw=True, use_det_path=True, use_lat_path=True,
		det_encoder_params=None, lat_encoder_params=None, decoder_params=None,
		sample_latent_post=True, sample_latent_prior=False):
		"""
		Args:
			in_shape (tuple): shape of the network's input feature
			context_size (int>0): context set size
			target_size (int>0): target set size
			out_size (int>0): size of the label vector
			det_encoder_params (dict): deterministic encoder hyperparameters
			lat_encoder_params (dict): latent encoder hyperparameters
			decoder_params (dict): decoder hyperparameters
		"""
		super().__init__()
		self.in_shape = in_shape
		self.out_size = out_size
		self.sample_latent_post = sample_latent_post
		self.sample_latent_prior = sample_latent_prior
		ft_params = ft_params or {}
		det_encoder_params = det_encoder_params or {}
		lat_encoder_params = lat_encoder_params or {}
		decoder_params = decoder_params or {}

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
			emb_shape = self.feat_transform.out_shape
		else:
			emb_shape = self.in_shape

		self.feat_norm_fn = NORM_MAPPING.get(fn_name, None)
		if (self.feat_norm_fn):
			self.context_feat_norm = self.feat_norm_fn(emb_shape, **fn_params)
			if (fn_split):
				self.target_feat_norm = self.feat_norm_fn(emb_shape, **fn_params)
			else:
				self.target_feat_norm = self.context_feat_norm

		dec_in_shape = [target_size, 0]
		if (use_raw):
			dec_in_shape[-1] += emb_shape[-1]

		self.det_encoder = None
		if (use_det_path):
			self.det_encoder = DetEncoder(emb_shape, context_size, target_size, self.out_size, **det_encoder_params)
			assert self.det_encoder.out_shape[0] == dec_in_shape[0]
			dec_in_shape[-1] += self.det_encoder.out_shape[-1]
			# print(f'{self.det_encoder.in_shape=}')
			# print(f'{self.det_encoder.out_shape=}')

		self.lat_encoder = None
		if (use_lat_path):
			self.lat_encoder = LatEncoder(emb_shape, out_size, **lat_encoder_params)
			dec_in_shape[-1] += self.lat_encoder.out_shape[-1]
			# print(f'{self.lat_encoder.in_shape=}')
			# print(f'{self.lat_encoder.out_shape=}')

		self.decoder = Decoder(tuple(dec_in_shape), target_size, self.out_size, use_raw, use_det_path, use_lat_path,
			self.det_encoder, self.lat_encoder, **decoder_params)
		# print(f'{self.decoder.in_shape=}')
		# print(f'{self.decoder.out_shape=}')
		self.out_shape = self.decoder.out_shape

	def forward(self, context_x, context_y, target_x, target_y=None):
		"""
		Propagate context and target through neural process network.

		Returns:
			latent prior, latent posterior, and output distributions
		"""
		if (is_valid(self.input_norm_fn)):
			context_x = self.context_input_norm(context_x)
			target_x = self.target_input_norm(target_x)

		if (is_valid(self.feat_transform_fn)):
			context_h = self.feat_transform(collapse_lower(context_x))
			target_h = self.feat_transform(collapse_lower(target_x))
			context_h = uncollapse_lower(context_h, context_x.shape[:2])
			target_h = uncollapse_lower(target_h, target_x.shape[:2])
		else:
			context_h, target_h = context_x, target_x

		# if (is_valid(self.feat_norm_fn)):
		# 	context_h = self.context_feat_norm(context_h)
		# 	target_h = self.target_feat_norm(target_h)

		det_rep = lat_rep = prior_dist = post_dist = None
		context_y = torch.atleast_3d(context_y)

		if (is_valid(self.det_encoder)):
			det_rep = self.det_encoder(context_h, context_y, target_h)

		if (is_valid(self.lat_encoder)):
			prior_dist = self.lat_encoder(context_h, context_y)

			if (is_valid(target_y)):
				# At training time:
				target_y = torch.atleast_3d(target_y)
				post_dist = self.lat_encoder(target_h, target_y)
				lat_rep = post_dist.rsample() if (self.sample_latent_post) \
					else post_dist.mean
			else:
				# At test time:
				post_dist = None
				lat_rep = prior_dist.rsample() if (self.sample_latent_prior) \
					else prior_dist.mean

		out_dist = self.decoder(det_rep, lat_rep, target_h)
		return prior_dist, post_dist, out_dist

