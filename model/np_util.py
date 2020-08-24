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

from common_util import is_valid
from model.common import dum
from model.model_util import log_prob_sigma, kl_loss_var, init_layer, get_padding, pyt_multihead_attention, FFN, TemporalConvNet
# Tensors are column-major, shaped as (batch, channel, sequence) unless otherwise specified
# Inspired by:
# * https://github.com/3springs/attentive-neural-processes/blob/master/neural_processes/models/neural_process/model.py
# * https://github.com/soobinseo/Attentive-Neural-Process/blob/master/module.py


# ********** HELPER MODULES **********
class DetConvEncoder(nn.Module):
	"""
	Attentive Neural Process (ANP) Convolutional (TCN) Deterministic Decoder.
	Takes in examples of shape (n, C, S), where
		n: batch size
		C: channels
		S: sequence

	XXX:
		* Target transform is a tcn
		* Context transform is a tcn
	"""
	def __init__(self, in_shape, embed_size=128, encoder_depth=3,
		encoder_kernel_size=3, encoder_dropout=0.0, label_size=1,
		sa_depth=2, sa_heads=8, sa_dropout=0.0,
		xa_depth=2, xa_heads=8, xa_dropout=0.0):
		"""
		Args:
			in_shape (tuple): input size as (C, S)
			embed_size (int>0): embedding size of tcn encoder and attention modules
			encoder_depth (int>0): tcn encoder depth
			encoder_kernel_size (int>0): convolution kernel size of tcn layers
			encoder_dropout (float>=0): tcn encoder dropout
			label_size (int>0): size of the label vector
			sa_depth (int>0): self-attention network depth
			sa_heads (int>0): self-attention heads
			sa_dropout (float>=0): self-attention dropout
			xa_depth (int>0): cross-attention network depth
			xa_heads (int>0): cross-attention heads
			xa_dropout (float>=0): cross-attention dropout
		"""
		super(DetConvEncoder, self).__init__()
		self.in_shape = in_shape
		self.label_size = label_size
		tcn_fn = partial(TemporalConvNet,
			block_channels=[[embed_size for _ in range(encoder_depth)]],
			pad_type='full', kernel_sizes=[encoder_kernel_size],
			global_dropout=encoder_dropout, no_dropout=None)
		self.target_transform = tcn_fn(in_shape)
		self.context_transform = tcn_fn(in_shape)
		self.input_encoder = tcn_fn((in_shape[0]+self.label_size, in_shape[1]))
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
		ANP Deterministic Convolutional Encoder forward Pass
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

class LatConvEncoder(nn.Module):
	"""
	Attentive Neural Process (ANP) Convolutional (TCN) Latent Decoder.

	XXX:
		* using linear layers for map_layer, mean, and logvar modules
		* taking mean over sequence dimension of encoder output
	"""
	def __init__(self, in_shape, embed_size=128, encoder_depth=3,
		encoder_kernel_size=3, encoder_dropout=0.0, latent_size=256, label_size=1,
		sa_depth=2, sa_heads=8, sa_dropout=0.0, min_std=.01, use_lvar=False):
		"""
		Args:
			in_shape (tuple): input size as (C, S)
			embed_size (int>0): embedding size of tcn encoder and self-attention modules
			encoder_depth (int>0): tcn encoder depth
			encoder_kernel_size (int>0): convolution kernel size of tcn layers
			encoder_dropout (float>=0): tcn encoder dropout
			latent_size (int>0): size of latent representation
			label_size (int>0): size of the label vector
			sa_depth (int>0): self-attention network depth
			sa_heads (int>0): self-attention heads
			sa_dropout (float>=0): self-attention dropout
			min_std (float): value used to limit latent distribution sigma
			use_lvar (bool): whether to use log domain variance clipping
		"""
		super(LatConvEncoder, self).__init__()
		self.in_shape = in_shape
		self.label_size = label_size
		self.input_encoder = TemporalConvNet(
			in_shape=(in_shape[0]+self.label_size, in_shape[1]),
			block_channels=[[embed_size for _ in range(encoder_depth)]],
			pad_type='full', kernel_sizes=[encoder_kernel_size],
			global_dropout=encoder_dropout, no_dropout=None)

		self.attention_fn = pyt_multihead_attention
		self.sa_W = nn.ModuleList([nn.MultiheadAttention(
			embed_size, sa_heads, dropout=sa_dropout, bias=True, add_bias_kv=False,
			add_zero_attn=False, kdim=None, vdim=None) for _ in range(sa_depth)])

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

		lat_dist = torch.distributions.Normal(lat_mean, lat_sigma)
		return lat_dist, lat_logvar

class ConvDecoder(nn.Module):
	"""
	ANP Convolutional Decoder.
	XXX:
		* target transform is a tcn
		* decoder is a tcn
		* mean and logsig layers are linear layers
		* decoder output is flattened to be able to send to mean and logsig layers
	"""
	def __init__(self, in_shape, embed_size=128, decoder_depth=3,
		decoder_kernel_size=3, decoder_dropout=0.0, label_size=1, dist_type='beta',
		encoder_embed_size=128, latent_size=256, min_std=.01, use_lvar=False):
		"""
		Args:
			in_shape (tuple): input size as (C, S)
			embed_size (int>0): embedding size of tcn decoder modules
			decoder_depth (int>0): tcn encoder depth
			decoder_kernel_size (int>0): convolution kernel size of tcn layers
			decoder_dropout (float>=0): tcn decoder dropout
			label_size (int>0): size of the label vector
			dist_type (str): output distribution, allowed values:
				* beta
				* normal
				* lognormal
				* gamma
				* dirichlet (add?)
				* bernoulli (add?)
			encoder_embed_size (int>0): embedding size of deterministic encoder
			latent_size (int>0): size of latent representation
			min_std (float): value used to limit output distribution sigma
			use_lvar (bool): whether to use log domain variance clipping
		"""
		super(ConvDecoder, self).__init__()
		self.in_shape = in_shape
		self.label_size = label_size
		tcn_fn = partial(TemporalConvNet,
			block_channels=[[embed_size for _ in range(decoder_depth)]],
			pad_type='full', kernel_sizes=[decoder_kernel_size],
			global_dropout=decoder_dropout, no_dropout=None)
		self.target_transform = tcn_fn(in_shape)

		self.decoder = tcn_fn((sum((encoder_embed_size, latent_size, embed_size)),
			self.target_transform.out_shape[1]))
		self.dist_type = dist_type
		self.dist_fn = {
			'beta': torch.distributions.Beta,
			'normal': torch.distributions.Normal,
			'lognormal': torch.distributions.LogNormal,
			'gamma': torch.distributions.Gamma
		}.get(self.dist_type, None)
		assert is_valid(self.dist_fn), 'dist_type must be a valid distribution type'

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

class ConvANP(nn.Module):
	"""
	Convolutional Attentive Neural Process Module
	"""
	def __init__(self, in_shape, label_size=1, out_dist_type='beta',
		context_in_target=False, use_lvar=False):
		"""
		Args:
			in_shape (tuple): input size as (C, S)
			label_size (int>0): size of the label vector
			out_dist_type (str): output distribution, allowed values:
				* beta
				* normal
				* lognormal
				* gamma
				* dirichlet (add?)
				* bernoulli (add?)
			use_lvar (bool): whether to use log domain variance clipping
		"""
		super(ConvANP, self).__init__()
		self.in_shape = in_shape
		self.label_size = label_size
		self.out_dist_type= out_dist_type
		self.det_encoder = DetConvEncoder(in_shape, label_size=label_size)
		self.lat_encoder = LatConvEncoder(in_shape, label_size=label_size)
		self.decoder = ConvDecoder(in_shape, label_size=label_size, \
			dist_type=self.out_dist_type)
		self.context_in_target = context_in_target
		self.use_lvar = use_lvar
		self.context_in_target = False
		self.mse = nn.MSELoss(reduction='none')
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
		y_pred = out_dist.rsample() if (train_mode) else out_dist.loc # XXX what if out_dist is a beta dist
		losses = None

		if (is_valid(target_y)):
			if (self.use_lvar):
				pass # custom log prob and kl div here
			else:
				# XXX loss when out_dist is a dist different from normal?
				logprob = out_dist.log_prob(target_y).mean(-1).unsqueeze(-1)
				kldiv = torch.distributions.kl_divergence(posterior_dist, prior_dist) \
					.mean(-1).unsqueeze(-1)
				# use kl beta factor (disentangled representation)?
				if (self.context_in_target):
					pass
			mse = self.mse(out_dist.loc, target_y) # TODO: switch to a classification type loss
			lossprob = -logprob

			# Weight loss nearer to prediction time?
			#weight = (torch.arange(lossprob.shape[1]) + 1).float().to(dev)[None, :]
			#lossprob_weighted = lossprob / torch.sqrt(weight)  # We want to  weight nearer stuff more
			losses = {
				'loss': (kldiv - logprob).mean(),
				'kldiv': kldiv.mean(),
				'mse': mse.mean(),
				'lossprob': lossprob.mean(),
				#'lossprob_weighted': lossprob_weighted.mean()
			}

		return y_pred, losses

