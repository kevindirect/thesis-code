"""
Neural Process utils
Kevin Patel
"""
import sys
import os
from operator import mul
from functools import reduce
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common_util import is_valid, isnt, fn_default_args, odd_only
from model.common import PYTORCH_ACT1D_LIST, PYTORCH_INIT_LIST
from model.model_util import log_prob_sigma, init_layer, get_padding, pt_multihead_attention, FFN, StackedTCN, TransposedTCN
# Tensors are column-major, shaped as (batch, channel, height, width) unless otherwise specified
# The in_shape and out_shape attributes of modules don't include the batch size
# Inspired by:
# * https://github.com/deepmind/neural-processes/blob/master/attentive_neural_process.ipynb
# * https://github.com/3springs/attentive-neural-processes/blob/master/neural_processes/models/neural_process/model.py
# * https://github.com/soobinseo/Attentive-Neural-Process/blob/master/module.py


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


# ********** HELPER MODULES **********
NP_TRANSFORMS_MAPPING = {
	'ffn': FFN,
	'stcn': StackedTCN,
	'ttcn': TransposedTCN
}


class DetEncoder(nn.Module):
	"""
	Attentive Neural Process (ANP) Deterministic Encoder.
	Takes in examples of shape (in_channels, in_height, in_width/sequence)

	Returns representation based on (Xc, yc) context points and Xt target points

	1. Xt -> Q or Transform Xt -> Q
	2. Xc -> K or Transform Xc -> K
	3. Transform (Xc, yc) -> V
	4. Run self attention on V
	5. Run cross attention on Q, K, V to get final representation
		(get weighted V based on similarity scores of Q to K)
	"""
	def __init__(self, in_shape, label_size, it_name='stcn', it_params=None,
		tt_name='stcn', tt_params=None, ct_name='stcn', ct_params=None,
		embed_size=128,
		sa_depth=2, sa_heads=8, sa_dropout=0.0,
		xa_depth=2, xa_heads=8, xa_dropout=0.0):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor,
				expects a shape (in_channels, in_height, in_width)
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

		self.cc_dim = 2 # add y to x as a constant row vector
		ie_shape = list(self.in_shape)
		ie_shape[self.cc_dim-1] += self.label_size
		self.input_encoder = NP_TRANSFORMS_MAPPING[it_name](ie_shape, **it_params)
		self.target_transform = NP_TRANSFORMS_MAPPING.get(tt_name, None)
		self.context_transform = NP_TRANSFORMS_MAPPING.get(ct_name, None)

		self.target_transform = self.target_transform and \
			self.target_transform(self.in_shape, **tt_params)
		self.context_transform = self.context_transform and \
			self.context_transform(self.in_shape, **ct_params)

		self.v_size = self.input_encoder.out_shape[0]
		self.q_size = self.target_transform.out_shape[0] if (self.target_transform) \
			else embed_size

		self.attention_fn = pt_multihead_attention
		self.sa_W = nn.ModuleList([nn.MultiheadAttention(
			self.v_size, sa_heads, dropout=sa_dropout, bias=True, add_bias_kv=False,
			add_zero_attn=False, kdim=None, vdim=None) for _ in range(sa_depth)])
		self.xa_W = nn.ModuleList([nn.MultiheadAttention(
			self.q_size, xa_heads, dropout=xa_dropout, bias=True, add_bias_kv=False,
			add_zero_attn=False, kdim=None, vdim=None) for _ in range(xa_depth)])
		self.out_shape = self.input_encoder.out_shape

	def forward(self, context_x, context_y, target_x):
		"""
		ANP Deterministic Encoder forward Pass
		Returns all representations, need to be aggregated to form r_*.
		"""
		# print('___:', context_x.shape, context_y.shape, target_x.shape)
		query = target_x if (isnt(self.target_transform)) \
			else self.target_transform(target_x)
		keys = context_x if (isnt(self.context_transform)) \
			else self.context_transform(context_x)
		xy = pt_concat_xy(context_x, context_y, self.label_size, dim=self.cc_dim)
		values = self.input_encoder(xy)
		query, keys, values = query.squeeze(), keys.squeeze(), values.squeeze()

		# Runs through stacked self and cross attention modules
		for W in self.sa_W:
			values, _ = self.attention_fn(W, values, values, values)
		for W in self.xa_W:
		# 	print('qkv:', query.shape, keys.shape, values.shape)
			query, _ = self.attention_fn(W, query, keys, values)
		return query

class LatEncoder(nn.Module):
	"""
	Attentive Neural Process (ANP) Latent Encoder.

	1. Transform (Xc, yc) -> enc
	2. Run self attention on enc
	3. Use aggregate representation of enc to parameterize a distribution (latent variable)
	
	XXX:
		* using linear layers for map_layer, alpha, and beta modules
		* taking mean over sequence dimension of encoder output
	"""
	def __init__(self, in_shape, label_size, it_name='stcn', it_params=None,
		latent_size=256, dist_type='normal', sa_depth=2, sa_heads=8, sa_dropout=0.0,
		min_std=.01, use_lvar=False):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor,
				expects a shape (in_channels, in_height, in_width)
			label_size (int>0): size of the label vector
			it_name (str): input transform name
			it_params (dict): input transform hyperparameters
			latent_size (int>0): size of latent representation
			dist_type (str): latent distribution, allowed values:
				* beta
				* normal
				* lognormal
			sa_depth (int>0): self-attention network depth
			sa_heads (int>0): self-attention heads
			sa_dropout (float>=0): self-attention dropout
			min_std (float): value used to limit latent distribution sigma
			use_lvar (bool): whether to use log domain variance clipping
		"""
		super().__init__()
		self.in_shape = in_shape
		self.label_size = label_size
		self.min_std, self.use_lvar = min_std, use_lvar
		if (isnt(it_params)):
			it_params = {}

		self.cc_dim = 2 # add y to x as a constant row vector
		ie_shape = list(self.in_shape)
		ie_shape[self.cc_dim-1] += self.label_size
		self.input_encoder = NP_TRANSFORMS_MAPPING[it_name](ie_shape, **it_params)
		self.attention_fn = pt_multihead_attention
		embed_size = self.input_encoder.out_shape[0]
		self.sa_W = nn.ModuleList([nn.MultiheadAttention(
			embed_size, sa_heads, dropout=sa_dropout, bias=True, add_bias_kv=False,
			add_zero_attn=False, kdim=None, vdim=None) for _ in range(sa_depth)])

		self.dist_type = dist_type
		self.dist_fn = {
			'beta': torch.distributions.Beta,
			'normal': torch.distributions.Normal,
			'lognormal': torch.distributions.LogNormal
		}.get(self.dist_type, None)
		assert is_valid(self.dist_fn), \
			'dist_type must be a valid latent distribution type'

		self.latent_size = latent_size
		self.map_layer = nn.Linear(embed_size, embed_size)	# maps attn to latent
		self.alpha = nn.Linear(embed_size, self.latent_size)	# latent variable param 1
		self.beta = nn.Linear(embed_size, self.latent_size)	# latent variable param 2

		# self.map_layer = nn.Conv1d(embed_size, embed_size, 1)
		# self.alpha = nn.Conv1d(embed_size, self.latent_size, 1)
		# self.beta = nn.Conv1d(embed_size, self.latent_size, 1)

		self.sig = None
		if (self.dist_type.endswith('normal')):
			self.sig = nn.LogSigmoid() if (self.use_lvar) else nn.Sigmoid()
		self.out_shape = (self.latent_size,)

	def forward(self, x, y):
		"""
		ANP Latent Convolutional Encoder forward Pass
		"""
		# xpd_y = y.reshape(y.shape[0], self.label_size, 1).float() \
		# 	.expand(x.shape[0], -1, x.shape[2]) # Matches y to x for non-channel dims
		# enc = self.input_encoder(torch.cat([x, xpd_y], dim=1))

		xy = pt_concat_xy(x, y, self.label_size, dim=self.cc_dim)
		enc = self.input_encoder(xy).squeeze()

		for W, _ in self.sa_W:
			enc = self.attention_fn(W, enc, enc, enc)

		sa_mean = torch.relu(self.map_layer(enc.mean(dim=2))) # Aggregate over sequence dim
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
					self.sig(lat_dist_beta * 0.5)
			lat_dist_beta = lat_dist_sigma

		lat_dist = self.dist_fn(lat_dist_alpha, lat_dist_beta)
		return lat_dist, lat_dist_beta

class Decoder(nn.Module):
	"""
	ANP Decoder.
	XXX:
		* mean and logsig layers are linear layers
		* decoder output is flattened to be able to send to mean and logsig layers
	"""
	def __init__(self, in_shape, label_size, use_det_path, det_encoder, lat_encoder,
		tt_name='stcn', tt_params=None, de_name='ttcn', de_params=None,
		dist_type='beta', min_std=.01, use_lvar=False):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor,
				expects a shape (in_channels, in_height, in_width)
			label_size (int>0): size of the label vector
			det_encoder (): deterministic encoder
			lat_encoder (): latent encoder
			tt_name (str): target transform name
			tt_params (dict): target transform hyperparameters
			de_name (str): decoder name
			de_params (dict): decoder hyperparameters
			dist_type (str): output distribution, allowed values:
				* bernoulli
				* categorical
				* beta
				* normal
				* lognormal
				* dirichlet (XXX add?)
			min_std (float): value used to limit output distribution sigma
			use_lvar (bool): whether to use log domain variance clipping
		"""
		super().__init__()
		self.in_shape = in_shape
		self.label_size = label_size
		self.min_std, self.use_lvar = min_std, use_lvar
		if (isnt(tt_params)):
			tt_params = {}
		if (isnt(de_params)):
			de_params = {}

		# det_embed_size = det_encoder_params.get('embed_size', \
		# 	fn_default_args(DetEncoder)['embed_size']) if (use_det_path) else 0
		# lat_latent_size = lat_encoder_params.get('latent_size', \
		# 	fn_default_args(LatEncoder)['latent_size'])
		# de_size = de_params.get('size', \
		# 	fn_default_args(NP_TRANSFORMS_MAPPING[de_name])['size'])
		# print(det_size, lat_size, de_size)

		self.target_transform = NP_TRANSFORMS_MAPPING.get(tt_name, None)
		self.target_transform = self.target_transform and \
			self.target_transform(self.in_shape, **tt_params)

		if (self.target_transform):
			xt_size = self.target_transform.out_shape[0]
			de_width = self.target_transform.out_shape[2]
		else:
			xt_size = self.in_shape[0]
			de_width = self.in_shape[2]

		de_chan = 2
		# assert(xt_size == lat_encoder.latent_size)
		if (use_det_path):
			de_chan = 3
			# assert(xt_size == det_encoder.q_size)
		de_height = xt_size
		self.decoder = NP_TRANSFORMS_MAPPING[de_name] \
			((de_chan, de_height, de_width), **de_params)

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
		self.alpha = nn.Linear(decoder_size, self.label_size) # primary out_dist parameter
		if (self.dist_type in ('bernoulli', 'categorical')):
			self.beta = None
			self.clamp = nn.Sigmoid()
		elif (self.dist_type in ('beta',)):
			self.beta = nn.Linear(decoder_size, self.label_size)
			self.clamp = nn.Softplus()
		elif (self.dist_type.endswith('normal')):
			self.beta = nn.Linear(decoder_size, self.label_size)
			self.clamp = torch.clamp if (self.use_lvar) else nn.Softplus()

		self.out_shape = (self.label_size,)

	def forward(self, det_rep, lat_rep, target_x):
		"""
		Args:
			det_rep (torch.tensor): deterministic representation (nullable)
			lat_rep (torch.tensor): global latent dist realization
			target_x (torch.tensor): target
		"""
		x = target_x if (isnt(self.target_transform)) else self.target_transform(target_x)
		x = x.squeeze()
		latent_rep = lat_rep.unsqueeze(2).expand(-1, -1, x.shape[-1]) #XXX
		decoder_inputs = [x, latent_rep]
		if (is_valid(det_rep)):
			decoder_inputs.append(det_rep)
		rep = torch.stack(decoder_inputs, dim=1)		# Multichannel 2d conv -> single channel 2d conv
		#rep = torch.cat(decoder_inputs, dim=1)#.unsqueeze(1)	# Single channel 2D conv
		decoded = self.decoder(rep)

		# For now we're flattening the decoded embedding
		out_dist_alpha = self.clamp(self.alpha(torch.flatten(decoded, start_dim=1, end_dim=-1)))

		if (self.dist_type in ('bernoulli', 'categorical')):
			out_dist = self.dist_fn(probs=out_dist_alpha)
		elif (self.dist_type in ('beta',)):
			out_dist_beta = self.clamp(self.beta(torch.flatten(decoded, start_dim=1, end_dim=-1)))
			out_dist = self.dist_fn(out_dist_alpha, out_dist_beta)
		elif (self.dist_type.endswith('normal')):
			out_dist_beta = self.clamp(self.beta(torch.flatten(decoded, start_dim=1, end_dim=-1)))
			if (self.use_lvar):
				out_dist_beta = self.clamp(out_dist_beta, math.log(self._min_std), \
					-math.log(self._min_std))
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
	def __init__(self, in_shape, label_size=1, use_det_path=True, det_encoder_params=None,
		lat_encoder_params=None, decoder_params=None, sample_latent=True,
		use_lvar=False, context_in_target=False):
		"""
		Args:
			in_shape (tuple): shape of the network's input tensor,
				expects a shape (in_channels, in_height, in_width)
			label_size (int>0): size of the label vector
			det_encoder_params (dict): deterministic encoder hyperparameters
			lat_encoder_params (dict): latent encoder hyperparameters
			decoder_params (dict): decoder hyperparameters
			sample_latent (bool): whether to sample latent dist or use EV
			use_lvar (bool):
			context_in_target (bool):
		"""
		super().__init__()
		self.in_shape = in_shape
		self.label_size = label_size
		self.use_lvar, self.context_in_target = use_lvar, context_in_target
		if (isnt(det_encoder_params)):
			det_encoder_params = {}
		if (isnt(lat_encoder_params)):
			lat_encoder_params = {}
		if (isnt(decoder_params)):
			decoder_params = {}

		self.det_encoder = DetEncoder(in_shape, label_size, **det_encoder_params) \
			if (use_det_path) else None
		self.lat_encoder = LatEncoder(in_shape, label_size, **lat_encoder_params)
		self.decoder = Decoder(in_shape, label_size, use_det_path, self.det_encoder,
			self.lat_encoder, **decoder_params)
		self.sample_latent = sample_latent
		self.bce = nn.BCEWithLogitsLoss()
		self.mae = nn.L1Loss()
		self.mse = nn.MSELoss()
		self.out_shape = self.decoder.out_shape

	def forward(self, context_x, context_y, target_x, target_y=None):
		"""
		Convenience method to propagate context and targets through networks,
		and sample the output distribution.
		"""
		prior, posterior, out = self.forward_net(context_x, context_y, target_x, \
			target_y=target_y)
		pred_y, losses = self.sample(prior, posterior, out, target_y=target_y)
		return pred_y, losses

	def forward_net(self, context_x, context_y, target_x, target_y=None):
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
		"""
		prior_dist, prior_beta = self.lat_encoder(context_x, context_y)
		posterior_dist, posterior_beta = None, None

		if (is_valid(target_y)):
			# At training time:
			posterior_dist, posterior_beta = self.lat_encoder(target_x, target_y)
			lat_rep = posterior_dist.rsample() if (self.sample_latent) \
				else posterior_dist.mean
			# if (train_mode):
			# 	lat_rep = posterior_dist.rsample() if (self.sample_latent) \
			# 		else posterior_dist.mean
			# else:
			# 	lat_rep = prior_dist.rsample() if (self.sample_latent) \
			# 		else prior_dist.mean
		else:
			# At test/inference time:
			lat_rep = prior_dist.rsample() if (self.sample_latent) else prior_dist.mean

		det_rep = self.det_encoder and self.det_encoder(context_x, context_y, target_x)
		out_dist = self.decoder(det_rep, lat_rep, target_x)
		return prior_dist, posterior_dist, out_dist

	def sample(self, prior_dist, posterior_dist, out_dist, target_y=None, \
		cast_precision=16):
		"""
		Sample neural proces output distribution to return prediction,
		calculate and return loss if a target label was passed in.

		Args:
			prior_dist ():
			posterior_dist ():
			out_dist ():
			target_y (torch.tensor):
			cast_precision (16|32|64):
		"""
		# At training time sample the output distribution, at test time use EV
		pred_y = out_dist.rsample() if (is_valid(target_y) and out_dist.has_rsample) \
			else out_dist.mean
		losses = None

		if (is_valid(target_y)):
			# At training time:
			if (self.use_lvar):
				pass # custom log prob and kl div here
			else:
				label_y = target_y
				if (type(out_dist).__name__ in ('Bernoulli', 'Beta', 'Normal')):
					ftype = {
						16: torch.float16,
						32: torch.float32,
						64: torch.float64
					}.get(cast_precision, 16)
					label_y = label_y.to(ftype)
					if (type(out_dist).__name__ in ('Beta',)):
						eps = 1e-3
						label_y = label_y.clamp(min=eps, max=1-eps)

				# print('target_y', label_y)
				# print('pred_y', pred_y)
				# # print('out_dist.mean', out_dist.mean)
				# # print('out_dist.log_prob(label_y)', out_dist.log_prob(label_y))
				# # for i in [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0]:
				# # 	print(str(i))
				# # 	print(out_dist.log_prob(torch.tensor([i], device='cuda')))
				# print('log pd:', out_dist.log_prob(label_y))
				# print('pd:', out_dist.log_prob(label_y).exp())

				# Here we get the likelihood of getting the ground truth in our
				# learned output distribution:
				logpd = out_dist.log_prob(label_y).mean(-1).unsqueeze(-1)

				# The KL divergence of the prior dist (conditioned on context) and
				# the posterior dist (conditioned on target during training).
				# The point of this term in the loss is to make sure the prior
				# and posterior aren't too different.
				kldiv = torch.distributions.kl_divergence(posterior_dist, prior_dist) \
					.mean(-1).unsqueeze(-1)
				# use kl beta factor (disentangled representation)?
				if (self.context_in_target):
					pass
			# Weight loss nearer to prediction time?
			# weight = (torch.arange(nll.shape[1]) + 1).float().to(dev)[None, :]
			# lossprob_weighted = nll / torch.sqrt(weight)  # We want to  weight nearer stuff more
			losses = {
				'loss': (kldiv - logpd).mean(),
				'logpd': logpd.mean(),
				'kldiv': kldiv.mean(),
				'bce': self.bce(out_dist.mean.squeeze(), label_y.squeeze()).mean(),
				'mae': self.mae(out_dist.mean.squeeze(), label_y.squeeze()).mean(),
				'mse': self.mse(out_dist.mean.squeeze(), label_y.squeeze()).mean()
				# 'lossprob_weighted': lossprob_weighted.mean()
			}

		return pred_y, losses

	@classmethod
	def suggest_params(cls, trial=None, num_classes=2, more_transforms=True):
		"""
		suggest model hyperparameters from an optuna trial object
		or return fixed default hyperparameters
		"""
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

			params = {
				'det_encoder_params': {
					'it_name': 'stcn', 'it_params': params_stcn,
					'tt_name': more_transforms and 'stcn', 'tt_params': params_stcn,
					'ct_name': more_transforms and 'stcn', 'ct_params': params_stcn,
					'embed_size': 128,
					'sa_depth': 2, 'sa_heads': 8, 'sa_dropout': 0.8,
					'xa_depth': 2, 'xa_heads': 8, 'xa_dropout': 0.8
				},
				'lat_encoder_params': {
					'it_name': 'stcn', 'it_params': params_stcn,
					'latent_size': 256,
					'dist_type': 'normal',
					'sa_depth': 2, 'sa_heads': 8, 'sa_dropout': 0.0,
					'min_std': .01, 'use_lvar': False
				},
				'decoder_params': {
					'tt_name': more_transforms and 'stcn', 'tt_params': params_stcn,
					'de_name': 'ttcn', 'de_params': params_ttcn,
					'dist_type': 'normal',
					'min_std': .01, 'use_lvar': False
				},
				'use_det_path': True,
				'sample_latent': True,
				'use_lvar': False,
				'context_in_target': False,
				'label_size': num_classes-1
			}
		else:

			params = {
				'det_encoder_params': {
					'it_name': 'stcn', 'it_params': params_stcn,
					'tt_name': more_transforms and 'stcn', 'tt_params': params_stcn,
					'ct_name': more_transforms and 'stcn', 'ct_params': params_stcn,
					'embed_size': 128,
					'sa_depth': 2, 'sa_heads': 8, 'sa_dropout': 0.8,
					'xa_depth': 2, 'xa_heads': 8, 'xa_dropout': 0.8
				},
				'lat_encoder_params': {
					'it_name': 'stcn', 'it_params': params_stcn,
					'embed_size': 128,
					'latent_size': 256,
					'dist_type': 'normal',
					'sa_depth': 2, 'sa_heads': 8, 'sa_dropout': 0.0,
					'min_std': .01, 'use_lvar': False
				},
				'decoder_params': {
					'tt_name': more_transforms and 'stcn', 'tt_params': params_stcn,
					'de_name': 'ttcn', 'de_params': params_ttcn,
					'dist_type': 'normal',
					'min_std': .01, 'use_lvar': False
				},
				'use_det_path': True,
				'sample_latent': True,
				'use_lvar': False,
				'context_in_target': False,
				'label_size': num_classes-1
			}

		return params

