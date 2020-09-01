"""
Neural Process utils
Kevin Patel
"""
import sys
import os
from operator import mul
from functools import partial
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common_util import is_valid, isnt, fn_default_args
from model.common import dum
from model.model_util import log_prob_sigma, init_layer, get_padding, pyt_multihead_attention, FFN, TemporalConvNet
# Tensors are column-major, shaped as (batch, channel, sequence) unless otherwise specified
# Inspired by:
# * https://github.com/3springs/attentive-neural-processes/blob/master/neural_processes/models/neural_process/model.py
# * https://github.com/soobinseo/Attentive-Neural-Process/blob/master/module.py


# ********** HELPER FUNCTIONS **********
class StackedTCN(TemporalConvNet):
	"""
	Wrapper Module for model_util.TemporalConvNet,
	creates a fixed width, single block TCN.
	"""
	def __init__(self, in_shape, size=128, depth=3, kernel_sizes=3,
		input_dropout=0.0, output_dropout=0.0, global_dropout=.5,
		global_dilation=True, block_act='elu', out_act='relu',
		block_init='xavier_uniform', out_init='xavier_uniform', pad_type='full'):
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
		"""
		dropouts = [None] * depth
		dropouts[0], dropouts[-1] = input_dropout, output_dropout
		super().__init__(in_shape, block_channels=[[size] * depth], num_blocks=1,
			kernel_sizes=kernel_sizes, dropouts=dropouts,
			global_dropout=global_dropout, global_dilation=global_dilation,
			block_act=block_act, out_act=out_act, block_init=block_init,
			out_init=out_init, pad_type=pad_type)

NP_TRANSFORMS_MAPPING = {
	'tcn': StackedTCN
}


# ********** HELPER MODULES **********
class DetEncoder(nn.Module):
	"""
	Attentive Neural Process (ANP) Deterministic Encoder.
	Takes in examples of shape (n, C, S), where
		n: batch size
		C: channels
		S: sequence
	"""
	def __init__(self, in_shape, label_size, it_name='tcn', it_params=None,
		tt_name='tcn', tt_params=None, ct_name='tcn', ct_params=None,
		embed_size=128,
		sa_depth=2, sa_heads=8, sa_dropout=0.0,
		xa_depth=2, xa_heads=8, xa_dropout=0.0):
		"""
		Args:
			in_shape (tuple): input size as (C, S)
			label_size (int>0): size of the label vector
			it_name (str): input transform name
			it_params (dict): input transform hyperparameters
			tt_name (str): target transform name
			tt_params (dict): target transform hyperparameters
			ct_name (str): context transform name
			ct_params (dict): context transform hyperparameters
			embed_size (int>0): embedding size of attention modules
			sa_depth (int>0): self-attention network depth
			sa_heads (int>0): self-attention heads
			sa_dropout (float>=0): self-attention dropout
			xa_depth (int>0): cross-attention network depth
			xa_heads (int>0): cross-attention heads
			xa_dropout (float>=0): cross-attention dropout
		"""
		super().__init__()
		self.in_shape = in_shape
		self.label_size = label_size
		if (isnt(it_params)):
			it_params = {}
		if (isnt(tt_params)):
			tt_params = {}
		if (isnt(ct_params)):
			ct_params = {}

		self.input_encoder = NP_TRANSFORMS_MAPPING[it_name] \
			((in_shape[0]+self.label_size, in_shape[1]), **it_params)
		self.target_transform = NP_TRANSFORMS_MAPPING[tt_name] \
			(self.in_shape, **tt_params)
		self.context_transform = NP_TRANSFORMS_MAPPING[ct_name] \
			(self.in_shape, **ct_params)

		self.attention_fn = pyt_multihead_attention
		self.sa_W = nn.ModuleList([nn.MultiheadAttention(
			embed_size, sa_heads, dropout=sa_dropout, bias=True, add_bias_kv=False,
			add_zero_attn=False, kdim=None, vdim=None) for _ in range(sa_depth)])
		self.xa_W = nn.ModuleList([nn.MultiheadAttention(
			embed_size, xa_heads, dropout=xa_dropout, bias=True, add_bias_kv=False,
			add_zero_attn=False, kdim=None, vdim=None) for _ in range(xa_depth)])
		self.out_shape = self.input_encoder.out_shape

	def forward(self, context_x, context_y, target_x):
		"""
		ANP Deterministic Encoder forward Pass
		Returns all representations, need to be aggregated to form r_*.
		"""
		query = self.target_transform(target_x)
		keys = self.context_transform(context_x)
		xpd_y = context_y.reshape(context_y.shape[0], self.label_size, 1).float() \
			.expand(context_x.shape[0], -1, context_x.shape[2]) # Matches y to x for non-channel dims
		values = self.input_encoder(torch.cat([context_x, xpd_y], dim=1))
		for W in self.sa_W:
			values = self.attention_fn(W, values, values, values)
		for W in self.xa_W:
			query = self.attention_fn(W, query, keys, values)
		return query

class LatEncoder(nn.Module):
	"""
	Attentive Neural Process (ANP) Latent Encoder.

	XXX:
		* using linear layers for map_layer, mean, and logvar modules
		* taking mean over sequence dimension of encoder output
	"""
	def __init__(self, in_shape, label_size, it_name='tcn', it_params=None,
		embed_size=128, latent_size=256, dist_type='normal',
		sa_depth=2, sa_heads=8, sa_dropout=0.0, min_std=.01, use_lvar=False):
		"""
		Args:
			in_shape (tuple): input size as (C, S)
			label_size (int>0): size of the label vector
			it_name (str): input transform name
			it_params (dict): input transform hyperparameters
			embed_size (int>0): embedding size of attention modules
			latent_size (int>0): size of latent representation
			dist_type (str): latent distribution, allowed values:
				* beta
				* normal
				* lognormal
				* gamma
			sa_depth (int>0): self-attention network depth
			sa_heads (int>0): self-attention heads
			sa_dropout (float>=0): self-attention dropout
			min_std (float): value used to limit latent distribution sigma
			use_lvar (bool): whether to use log domain variance clipping
		"""
		super().__init__()
		self.in_shape = in_shape
		self.label_size = label_size
		if (isnt(it_params)):
			it_params = {}

		self.input_encoder = NP_TRANSFORMS_MAPPING[it_name] \
			((in_shape[0]+self.label_size, in_shape[1]), **it_params)
		self.attention_fn = pyt_multihead_attention
		self.sa_W = nn.ModuleList([nn.MultiheadAttention(
			embed_size, sa_heads, dropout=sa_dropout, bias=True, add_bias_kv=False,
			add_zero_attn=False, kdim=None, vdim=None) for _ in range(sa_depth)])

		self.dist_type = dist_type
		self.dist_fn = {
			'beta': torch.distributions.Beta,
			'normal': torch.distributions.Normal,
			'lognormal': torch.distributions.LogNormal,
			'gamma': torch.distributions.Gamma
		}.get(self.dist_type, None)
		assert is_valid(self.dist_fn), \
			'dist_type must be a valid latent distribution type'

		self.map_layer = nn.Linear(embed_size, embed_size)
		self.mean = nn.Linear(embed_size, latent_size)
		self.logvar = nn.Linear(embed_size, latent_size)

		# self.map_layer = nn.Conv1d(embed_size, embed_size, 1)
		# self.mean = nn.Conv1d(embed_size, latent_size, 1)
		# self.logvar = nn.Conv1d(embed_size, latent_size, 1)

		self.min_std = min_std
		self.use_lvar = use_lvar
		self.sig = nn.LogSigmoid() if (self.use_lvar) else nn.Sigmoid()
		self.out_shape = (latent_size,)

	def forward(self, x, y):
		"""
		ANP Latent Convolutional Encoder forward Pass
		"""
		xpd_y = y.reshape(y.shape[0], self.label_size, 1).float() \
			.expand(x.shape[0], -1, x.shape[2]) # Matches y to x for non-channel dims
		enc = self.input_encoder(torch.cat([x, xpd_y], dim=1))
		for W in self.sa_W:
			enc = self.attention_fn(W, enc, enc, enc)
		sa_mean = torch.relu(self.map_layer(enc.mean(dim=2))) # Average over sequence dim
		lat_mean = self.mean(sa_mean)
		lat_logvar = self.logvar(sa_mean)

		if (self.use_lvar):
			# Variance clipping in the log domain (should be more stable)
			lat_logvar = self.sig(lat_logvar)
			lat_logvar = torch.clamp(lat_logvar, np.log(self.min_std), \
				-np.log(self.min_std))
			lat_sigma = torch.exp(0.5 * lat_logvar)
		else:
			# Simple variance clipping (from deep mind repo)
			lat_sigma = self.min_std + (1 - self.min_std) * self.sig(lat_logvar * 0.5)

		lat_dist = self.dist_fn(lat_mean, lat_sigma)
		return lat_dist, lat_logvar

class Decoder(nn.Module):
	"""
	ANP Decoder.
	XXX:
		* mean and logsig layers are linear layers
		* decoder output is flattened to be able to send to mean and logsig layers
	"""
	def __init__(self, in_shape, label_size, det_encoder_params, lat_encoder_params,
		tt_name='tcn', tt_params=None, de_name='tcn', de_params=None, embed_size=128,
		dist_type='normal', min_std=.01, use_lvar=False):
		"""
		Args:
			in_shape (tuple): input size as (C, S)
			label_size (int>0): size of the label vector
			det_encoder_params (dict): deterministic encoder hyperparameters
			lat_encoder_params (dict): latent encoder hyperparameters
			tt_name (str): target transform name
			tt_params (dict): target transform hyperparameters
			de_name (str): decoder name
			de_params (dict): decoder hyperparameters
			embed_size (int>0): embedding size of tcn decoder modules
			dist_type (str): output distribution, allowed values:
				* beta
				* normal
				* lognormal
				* gamma
				* dirichlet (add?)
				* bernoulli (add?)
			min_std (float): value used to limit output distribution sigma
			use_lvar (bool): whether to use log domain variance clipping
		"""
		super().__init__()
		self.in_shape = in_shape
		self.label_size = label_size
		if (isnt(tt_params)):
			tt_params = {}
		if (isnt(de_params)):
			de_params = {}

		det_embed_size = det_encoder_params.get('embed_size', \
			fn_default_args(DetEncoder)['embed_size'])
		lat_latent_size = lat_encoder_params.get('latent_size', \
			fn_default_args(LatEncoder)['latent_size'])
		de_size = de_params.get('size', \
			fn_default_args(NP_TRANSFORMS_MAPPING[de_name])['size'])
		decoder_channels = sum((det_embed_size, lat_latent_size, de_size))

		self.target_transform = NP_TRANSFORMS_MAPPING[tt_name] \
			(self.in_shape, **tt_params)
		self.decoder = NP_TRANSFORMS_MAPPING[de_name] \
			((decoder_channels, self.target_transform.out_shape[1]), **de_params)

		self.dist_type = dist_type
		self.dist_fn = {
			'beta': torch.distributions.Beta,
			'normal': torch.distributions.Normal,
			'lognormal': torch.distributions.LogNormal,
			'gamma': torch.distributions.Gamma
		}.get(self.dist_type, None)
		assert is_valid(self.dist_fn), \
			'dist_type must be a valid output distribution type'

		# Interpretation of alpha and beta depends on the dist_type
		self.alpha = nn.Linear(mul(*self.decoder.out_shape), self.label_size)
		self.beta = nn.Linear(mul(*self.decoder.out_shape), self.label_size)
		self.min_std = min_std
		self.use_lvar = use_lvar
		self.softplus = None if (self.use_lvar or not self.dist_type.endswith('normal')) \
			else nn.Softplus()
		self.out_shape = (self.label_size,)

	def forward(self, det_rep, lat_rep, target_x):
		"""
		Args:
			det_rep (torch.tensor): deterministic representation
			lat_rep (torch.tensor): global latent dist realization
			target_x (torch.tensor): target
		"""
		x = self.target_transform(target_x)
		decoded = self.decoder(torch.cat([det_rep, lat_rep.unsqueeze(2) \
			.expand(-1, -1, x.shape[2]), x], dim=1))

		# For now we're flattening the decoded embedding
		out_dist_alpha = self.alpha(torch.flatten(decoded, start_dim=1, end_dim=-1))
		out_dist_beta = self.beta(torch.flatten(decoded, start_dim=1, end_dim=-1))

		if (self.dist_type.endswith('normal')):
			if (self.use_lvar):
				out_dist_beta = torch.clamp(out_dist_beta, math.log(self._min_std), \
					-math.log(self._min_std))
				out_dist_sigma = torch.exp(out_dist_beta)
			else:
				out_dist_sigma = self.min_std + (1 - self.min_std) * self.softplus(out_dist_beta)
			out_dist_beta = out_dist_sigma # Bounded or clamped variance

		return self.dist_fn(out_dist_alpha, out_dist_beta)


# ********** MODEL MODULES **********
class AttentiveNP(nn.Module):
	"""
	Attentive Neural Process Module
	"""
	def __init__(self, in_shape, label_size=1, det_encoder_params=None,
		lat_encoder_params=None, decoder_params=None, use_lvar=False,
		context_in_target=False):
		"""
		Args:
			in_shape (tuple): input size as (C, S)
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
		if (isnt(det_encoder_params)):
			det_encoder_params = {}
		if (isnt(lat_encoder_params)):
			lat_encoder_params = {}
		if (isnt(decoder_params)):
			decoder_params = {}

		self.det_encoder = DetEncoder(in_shape, label_size, **det_encoder_params)
		self.lat_encoder = LatEncoder(in_shape, label_size, **lat_encoder_params)
		self.decoder = Decoder(in_shape, label_size, det_encoder_params,
			lat_encoder_params, **decoder_params)
		self.use_lvar, self.context_in_target = use_lvar, context_in_target
		self.out_shape = self.decoder.out_shape

	def forward(self, context_x, context_y, target_x, target_y=None, \
		train_mode=False, sample_latent=True):
		"""
		Convenience method to propagate context and targets through networks,
		and sample the output distribution.
		"""
		prior, posterior, out = self.forward_net(context_x, context_y, target_x, \
			target_y, train_mode=train_mode, sample_latent=sample_latent)
		y_pred, losses = self.sample(prior, posterior, out, target_y, train_mode)
		return y_pred, losses

	def forward_net(self, context_x, context_y, target_x, target_y=None, \
		train_mode=False, sample_latent=True):
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
			sample_latent (bool): if False, use sample mean instead of sampling latent
		"""
		det_rep = self.det_encoder(context_x, context_y, target_x)
		prior_dist, prior_logvar = self.lat_encoder(context_x, context_y)

		if (is_valid(target_y)):
			# Training:
			posterior_dist, posterior_logvar = self.lat_encoder(target_x, target_y)
			if (train_mode):
				lat_rep = posterior_dist.rsample() if (sample_latent) else posterior_dist.loc
			else:
				lat_rep = prior_dist.rsample() if (sample_latent) else prior_dist.loc
		else:
			# Generation:
			lat_rep = prior_dist.rsample() if (sample_latent) else prior_dist.loc

		out_dist = self.decoder(det_rep, lat_rep, target_x)
		return prior_dist, posterior_dist, out_dist

	def sample(self, prior_dist, posterior_dist, out_dist, target_y=None, \
		train_mode=False):
		"""
		Sample neural proces output distribution to return prediction,
		calculate and return loss if a target label was passed in.

		Args:
			prior_dist ():
			posterior_dist ():
			out_dist ():
			target_y (torch.tensor):
			train_mode (bool): whether the model is in training or not.
				If in training, the model will sample the output distribution
				instead of using its first moment.
		"""
		y_pred = out_dist.rsample() if (train_mode) else out_dist.loc # XXX ?
		losses = None

		if (is_valid(target_y)):
			if (self.use_lvar):
				pass # custom log prob and kl div here
			else:
				nll = -out_dist.log_prob(target_y).mean(-1).unsqueeze(-1)
				kldiv = torch.distributions.kl_divergence(posterior_dist, prior_dist) \
					.mean(-1).unsqueeze(-1)
				# use kl beta factor (disentangled representation)?
				if (self.context_in_target):
					pass
			# mse = self.mse(out_dist.loc, target_y) # TODO: switch to a classifier loss (BCE?)

			# Weight loss nearer to prediction time?
			# weight = (torch.arange(nll.shape[1]) + 1).float().to(dev)[None, :]
			# lossprob_weighted = nll / torch.sqrt(weight)  # We want to  weight nearer stuff more
			losses = {
				'loss': (nll + kldiv).mean(),
				'nll': nll.mean(),
				'kldiv': kldiv.mean(),
				# 'mse': mse.mean(),
				# 'lossprob_weighted': lossprob_weighted.mean()
			}

		return y_pred, losses

