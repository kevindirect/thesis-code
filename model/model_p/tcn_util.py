"""
Kevin Patel

Adapted from code by locuslab and flrngel
Source: https://github.com/locuslab/TCN, https://github.com/flrngel/TCN-with-attention
"""
import sys
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from common_util import MODEL_DIR
from model.common import PYTORCH_MODELS_DIR, ERROR_CODE, TEST_RATIO, VAL_RATIO


class Chomp1d(nn.Module):
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous() # XXX - chomps off (kernel_size-1)*dilation off right end


class TemporalBlock(nn.Module):
	"""
	TCN Block Class

	DC: Dilated Casual Convolution
	WN: Weight Normalization
	RU: ReLu
	DO: Dropout

	----| DC |-->| WN |-->| RU |-->| DO |-->| DC |-->| WN |-->| RU |-->| DO |--(+)---->
		|-------------------------|1x1 Conv (optional)|-------------------------|
	"""
	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
		super(TemporalBlock, self).__init__()
		self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
		self.chomp1 = Chomp1d(padding)
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)

		self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
		self.chomp2 = Chomp1d(padding)
		self.relu2 = nn.ReLU()
		self.dropout2 = nn.Dropout(dropout)

		self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
		self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if (n_inputs != n_outputs) else None
		self.relu = nn.ReLU()
		self.init_weights()

	def init_weights(self, init_method='xavier_uniform'):
		"""
		Initialize convolutional layer weights.
		TCN networks are influenced heavily by weight initialization.
		"""
		if (init_method == 'normal'):
			self.conv1.weight.data.normal_(0, 0.01)
			self.conv2.weight.data.normal_(0, 0.01)
			if (self.downsample is not None):
				self.downsample.weight.data.normal_(0, 0.01)

		elif (init_method in ('xavier_uniform', 'glorot_uniform')):
			nn.init.xavier_uniform_(self.conv1.weight, gain=np.sqrt(2))
			nn.init.xavier_uniform_(self.conv2.weight, gain=np.sqrt(2))
			if (self.downsample is not None):
				nn.init.xavier_uniform_(self.downsample.weight, gain=np.sqrt(2))

	def forward(self, x):
		out = self.net(x)
		res = x if (self.downsample is None) else self.downsample(x)
		return self.relu(out + res)


class AttentionBlock(nn.Module):
	"""
	An attention mechanism similar to Vaswani et al (2017)
	The input of the AttentionBlock is 'BxTxD' where 'B' is the input
	minibatch size, 'T' is the length of the sequence 'D' is the dimensions of each feature.
	The output of the AttentionBlock is 'BxTx(D+V)' where 'V' is the size of the attention values.

	Arguments:
		dims (int): the number of dimensions (or channels) of each element in the input sequence
		k_size (int): the size of the attention keys
		v_size (int): the size of the attention values
		seq_len (int): the length of the input and output sequences
	"""
	def __init__(self, dims, k_size, v_size, seq_len=None):
		super(AttentionBlock, self).__init__()
		self.key_layer = nn.Linear(dims, k_size)
		self.query_layer = nn.Linear(dims, k_size)
		self.value_layer = nn.Linear(dims, v_size)
		self.sqrt_k = math.sqrt(k_size)

	def forward(self, minibatch):
		keys, queries, values = self.key_layer(minibatch), self.query_layer(minibatch), self.value_layer(minibatch)

		logits = torch.bmm(queries, keys.transpose(2,1))  # Batch matrix multiplication of key and query layers

		# Use numpy triu because you can't do 3D triu with PyTorch
		# TODO: using float32 here might break for non FloatTensor inputs.
		# Should update this later to use numpy/PyTorch types of the input.
		mask = torch.from_numpy(np.triu(np.ones(logits.size()), k=1).astype('uint8')) # 3D triangle mask

		# do masked_fill_ on data rather than Variable because PyTorch doesn't
		# support masked_fill_ w/-inf directly on Variables for some reason.
		logits.data.masked_fill_(mask, float('-inf'))
		probs = F.softmax(logits, dim=1) / self.sqrt_k
		read = torch.bmm(probs, values)
		return minibatch + read


class TemporalConvNet(nn.Module):
	"""
	Temporal Conv Net with Optional Attention Blocks.
	This class can be used to build classifiers and regressors.
	"""
	def __init__(self, num_input_channels, channels, kernel_size, stride, dropout, attention, max_attn_len):
		super(TemporalConvNet, self).__init__()
		layers = []
		num_levels = len(channels)
		for i in range(num_levels):
			dilation_size = 2 ** i
			in_channels = num_input_channels if (i == 0) else channels[i-1]
			out_channels = channels[i]
			layers += [TemporalBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation_size,
									padding=(kernel_size-1) * dilation_size, dropout=dropout)]
			if (attention):
				layers += [AttentionBlock(max_attn_len, max_attn_len, max_attn_len)]
		self.network = nn.Sequential(*layers)

	def forward(self, x):
		return self.network(x)


class TCN_Classifier(nn.Module):
	"""
	TCN Based Classifier Network

	Args:
		num_input_channels (int): number of input channels
		channels (list): list of output channel sizes in order from first to last
		num_outputs (int): number of outputs, usually the number of classes to predict (defaults to binary case)
		kernel_size (int > 1): CNN kernel size
		stride (int > 0): CNN kernel's stride 
		dropout (float [0, 1]): dropout probability, probability of an element to be zeroed during training
		attention (bool): whether or not to include attention block after each tcn block
		max_attn_len (int > 0): max length of attention (only relevant if attention is set to True)
	"""
	def __init__(self, num_input_channels, channels, num_outputs=1, kernel_size=2, stride=1, dropout=0.2, attention=False, max_attn_len=80):
		super(TCN_Classifier, self).__init__()
		self.tcn = TemporalConvNet(num_input_channels, channels, kernel_size=kernel_size, stride=stride, dropout=dropout, attention=attention, max_attn_len=max_attn_len)
		if (attention):
			self.linear = nn.Linear(max_attn_len, num_outputs) # TODO - verify correctness of using max_attn_len as input size to output layer
		else:
			self.linear = nn.Linear(channels[-1], num_outputs)
		self.output = nn.LogSoftmax(dim=1)

	def forward(self, x):
		"""
		Input must have have shape (N, C_in, L_in) where
			N: number of batches
			C_in: number of input channels
			L_in: length of input sequence
		"""
		out_embedding = self.tcn(x)
		# out = self.linear(out_embedding[:, :, -1])
		# out = self.linear(out_embedding).double()
		out = self.linear(out_embedding[:, :, -1])
		return self.output(out)
