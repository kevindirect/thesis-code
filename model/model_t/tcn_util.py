"""
Kevin Patel

Adapted from code by Weiping Song
Source: https://github.com/Songweiping/TCN-TF

Author: Weiping Song
Time: June 8, 2018

Reference: 
	weight norm paper: https://arxiv.org/pdf/1602.07868.pdf
	https://github.com/llan-ml/weightnorm
	https://github.com/openai/weightnorm
"""
import sys
import os
from os import sep
import logging

from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl
import tensorflow as tf

from common_util import MODEL_DIR
from model.common import TENSORFLOW_MODEL_DIR


class _WNConv(convolutional_layers._Conv):
	"""
	Weight Normalized Convolutional Layer
	"""
	def __init__(self, *args, **kwargs):
		self.weight_norm = kwargs.pop('weight_norm')
		super(_WNConv, self).__init__(*args, **kwargs)


	def build(self, input_shape):
		input_shape = tensor_shape.TensorShape(input_shape)
		if self.data_format == 'channels_first':
			channel_axis = 1
		else:
			channel_axis = -1

		if input_shape[channel_axis].value is None:
			raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
		input_dim = input_shape[channel_axis].value
		kernel_shape = self.kernel_size + (input_dim, self.filters)

		kernel = self.add_variable(name='kernel',
									shape=kernel_shape,
									initializer=self.kernel_initializer,
									regularizer=self.kernel_regularizer,
									constraint=self.kernel_constraint,
									trainable=True,
									dtype=self.dtype)
		# weight normalization 
		if self.weight_norm:
			g = self.add_variable(name='wn/g',
									shape=(self.filters,),
									initializer=init_ops.ones_initializer(),
									dtype=kernel.dtype,
									trainable=True)
			self.kernel = tf.reshape(g,[1,1,self.filters]) * nn_impl.l2_normalize(kernel, [0, 1]) 
		else:
			self.kernel = kernel
		
		if self.use_bias:
			self.bias = self.add_variable(name='bias',
											shape=(self.filters,),
											initializer=self.bias_initializer,
											regularizer=self.bias_regularizer,
											constraint=self.bias_constraint,
											trainable=True,
											dtype=self.dtype)
		else:
			self.bias = None
		self.input_spec = base.InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
		self._convolution_op = nn_ops.Convolution(
			input_shape,
			filter_shape=self.kernel.get_shape(),
			dilation_rate=self.dilation_rate,
			strides=self.strides,
			padding=self.padding.upper(),
			data_format=utils.convert_data_format(self.data_format, self.rank + 2))
		self.built = True


class WNConv1D(_WNConv):
	"""
	Top level Weight Normalized 1D Convolutional Class.
	"""
	def __init__(self, filters,
				kernel_size,
				strides=1,
				padding='valid',
				data_format='channels_last',
				dilation_rate=1,
				activation=None,
				weight_norm=False,
				use_bias=True,
				kernel_initializer=None,
				bias_initializer=init_ops.zeros_initializer(),
				kernel_regularizer=None,
				bias_regularizer=None,
				activity_regularizer=None,
				kernel_constraint=None,
				bias_constraint=None,
				trainable=True,
				name=None,
				**kwargs):
		super(WNConv1D, self).__init__(
			rank=1,
			filters=filters,
			kernel_size=kernel_size,
			strides=strides,
			padding=padding,
			data_format=data_format,
			dilation_rate=dilation_rate,
			activation=activation,
			weight_norm=weight_norm,
			use_bias=use_bias,
			kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer,
			kernel_regularizer=kernel_regularizer,
			bias_regularizer=bias_regularizer,
			activity_regularizer=activity_regularizer,
			kernel_constraint=kernel_constraint,
			bias_constraint=bias_constraint,
			trainable=trainable,
			name=name, **kwargs)

def wnconv1d(inputs,
			filters,
			kernel_size,
			strides=1,
			padding='valid',
			data_format='channels_last',
			dilation_rate=1,
			activation=None,
			weight_norm=False,
			use_bias=True,
			kernel_initializer=None,
			bias_initializer=init_ops.zeros_initializer(),
			kernel_regularizer=None,
			bias_regularizer=None,
			activity_regularizer=None,
			kernel_constraint=None,
			bias_constraint=None,
			trainable=True,
			name=None,
			reuse=None):
	layer = WNConv1D(
		filters=filters,
		kernel_size=kernel_size,
		strides=strides,
		padding=padding,
		data_format=data_format,
		dilation_rate=dilation_rate,
		activation=activation,
		weight_norm=weight_norm,
		use_bias=use_bias,
		kernel_initializer=kernel_initializer,
		bias_initializer=bias_initializer,
		kernel_regularizer=kernel_regularizer,
		bias_regularizer=bias_regularizer,
		activity_regularizer=activity_regularizer,
		kernel_constraint=kernel_constraint,
		bias_constraint=bias_constraint,
		trainable=trainable,
		name=name,
		dtype=inputs.dtype.base_dtype,
		_reuse=reuse,
		_scope=name) 
	
	return layer.apply(inputs)


class TemporalConvNet(object):
	"""
	TCN (Temporal Convolutional Net) with weight normalization, by Weiping Song.
	Source: https://github.com/Songweiping/TCN-TF
	"""
	def __init__(self, num_channels, stride=1, kernel_size=2, dropout=0.2):
		"""
		"""
		self.kernel_size = kernel_size
		self.stride = stride
		self.num_levels = len(num_channels)
		self.num_channels = num_channels
		self.dropout = dropout

	def __call__(self, inputs):
		"""
		Creates the Temporal CNN with exponential dilation factor.
		"""
		inputs_shape = inputs.get_shape().as_list()
		outputs = [inputs]

		for i in range(self.num_levels):
			dilation_size = 2 ** i
			in_channels = inputs_shape[-1] if i == 0 else self.num_channels[i-1]
			out_channels = self.num_channels[i]
			output = self._TemporalBlock(value=outputs[-1], n_inputs=in_channels, n_outputs=out_channels, kernel_size=self.kernel_size, stride=self.stride,
										dilation=dilation_size, padding=(self.kernel_size-1)*dilation_size, dropout=self.dropout, level=i)
			outputs.append(output)

		return outputs[-1]


	def _TemporalBlock(self, value, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, level=0):
		"""
		TCN Block

		--|Dilated Causal Conv|-->|WeightNorm|-->|ReLU|-->|Dropout|-->|Dilated Causal Conv|-->|WeightNorm|-->|ReLU|-->|Dropout|--(+)-->
		  |-------------------------------------------------|1x1 Conv (optional)|-------------------------------------------------|
		"""
		padded_value1 = tf.pad(value, [[0,0], [padding,0], [0,0]])
		self.conv1 = wnconv1d(inputs=padded_value1,
								filters=n_outputs,
								kernel_size=kernel_size,
								strides=stride,
								padding='valid',
								dilation_rate=dilation,
								activation=None,
								weight_norm=True, #default is false.
								kernel_initializer=tf.random_normal_initializer(0, 0.01),
								bias_initializer=tf.zeros_initializer(),
								name='layer{lvl}_conv1'.format(lvl=level))
		self.output1 = tf.nn.dropout(tf.nn.relu(self.conv1), keep_prob=1-dropout)

		padded_value2 = tf.pad(self.output1, [[0,0], [padding,0], [0,0]])
		self.conv2 = wnconv1d(inputs=padded_value2,
								filters=n_outputs,
								kernel_size=kernel_size,
								strides=stride,
								padding='valid',
								dilation_rate=dilation,
								activation=None,
								weight_norm=True, #default is False.
								kernel_initializer=tf.random_normal_initializer(0, 0.01),
								bias_initializer=tf.zeros_initializer(),
								name='layer{lvl}_conv2'.format(lvl=level))
		self.output2 = tf.nn.dropout(tf.nn.relu(self.conv2), keep_prob=1-dropout)

		if (n_inputs != n_outputs):
			res_x = tf.layers.conv1d(inputs=value,
									filters=n_outputs,
									kernel_size=1,
									activation=None,
									kernel_initializer=tf.random_normal_initializer(0, 0.01),
									bias_initializer=tf.zeros_initializer(),
									name='layer{lvl}_conv'.format(lvl=level))
		else:
			res_x = value

		return tf.nn.relu(res_x + self.output2)
