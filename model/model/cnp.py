"""
Docstring
Kevin Patel
"""

import numpy as np
import pandas as pd
import sys
import os


class DeterministicEncoder(object):
	"""
	The Encoder.

	Adapted from https://github.com/deepmind/conditional-neural-process
	"""
	def __init__(self, output_sizes):
		"""
		CNP encoder.

		Args:
			output_sizes: An iterable containing the output sizes of the encoding MLP.
		"""
		self._output_sizes = output_sizes

	def __call__(self, context_x, context_y, num_context_points):
		"""
		Encodes the inputs into one representation.

		Args:
			context_x: Tensor of size bs x observations x m_ch. For this 1D regression ask this corresponds to the x-values.
			context_y: Tensor of size bs x observations x d_ch. For this 1D regression ask this corresponds to the y-values.
			num_context_points: A tensor containing a single scalar that indicates the number of context_points provided in this iteration.

		Returns:
			representation: The encoded representation averaged over all context points.
		"""
		# Concatenate x and y along the filter axes
		encoder_input = tf.concat([context_x, context_y], axis=-1)

		# Get the shapes of the input and reshape to parallelise across observations
		batch_size, _, filter_size = encoder_input.shape.as_list()
		hidden = tf.reshape(encoder_input, (batch_size * num_context_points, -1))
		hidden.set_shape((None, filter_size))

		# Pass through MLP
		with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
			for i, size in enumerate(self._output_sizes[:-1]):
				hidden = tf.nn.relu(tf.layers.dense(hidden, size, name="Encoder_layer_{}".format(i)))

			# Last layer without a ReLu
			hidden = tf.layers.dense(hidden, self._output_sizes[-1], name="Encoder_layer_{}".format(i + 1))

		# Bring back into original shape
		hidden = tf.reshape(hidden, (batch_size, num_context_points, size))

		# Aggregator: take the mean over all points
		representation = tf.reduce_mean(hidden, axis=1)

		return representation


class DeterministicDecoder(object):
	"""
	The Decoder.

	Adapted from https://github.com/deepmind/conditional-neural-process
	"""

	def __init__(self, output_sizes):
		"""
		CNP decoder.

		Args:
			output_sizes: An iterable containing the output sizes of the decoder MLP as defined in `basic.Linear`.
		"""
		self._output_sizes = output_sizes

	def __call__(self, representation, target_x, num_total_points):
		"""
		Decodes the individual targets.

		Args:
			representation: The encoded representation of the context
			target_x: The x locations for the target query
			num_total_points: The number of target points.

		Returns:
			dist: A multivariate Gaussian over the target points.
			mu: The mean of the multivariate Gaussian.
			sigma: The standard deviation of the multivariate Gaussian.
		"""

		# Concatenate the representation and the target_x
		representation = tf.tile(tf.expand_dims(representation, axis=1), [1, num_total_points, 1])
		input = tf.concat([representation, target_x], axis=-1)

		# Get the shapes of the input and reshape to parallelise across observations
		batch_size, _, filter_size = input.shape.as_list()
		hidden = tf.reshape(input, (batch_size * num_total_points, -1))
		hidden.set_shape((None, filter_size))

		# Pass through MLP
		with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
			for i, size in enumerate(self._output_sizes[:-1]):
				hidden = tf.nn.relu(tf.layers.dense(hidden, size, name="Decoder_layer_{}".format(i)))

			# Last layer without a ReLu
			hidden = tf.layers.dense(hidden, self._output_sizes[-1], name="Decoder_layer_{}".format(i + 1))

		# Bring back into original shape
		hidden = tf.reshape(hidden, (batch_size, num_total_points, -1))

		# Get the mean an the variance
		mu, log_sigma = tf.split(hidden, 2, axis=-1)

		# Bound the variance
		sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

		# Get the distribution
		dist = tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

		return dist, mu, sigma


class DeterministicModel(object):
	"""
	The CNP model.

	Adapted from https://github.com/deepmind/conditional-neural-process
	"""

	def __init__(self, encoder_output_sizes, decoder_output_sizes):
		"""
		Initialises the model.

		Args:
			encoder_output_sizes: An iterable containing the sizes of hidden layers of the encoder. The last one is the size of the representation r.
			decoder_output_sizes: An iterable containing the sizes of hidden layers of the decoder. The last element should correspond to the dimension of
				the y * 2 (it encodes both mean and variance concatenated)
		"""
		self._encoder = DeterministicEncoder(encoder_output_sizes)
		self._decoder = DeterministicDecoder(decoder_output_sizes)

	def __call__(self, query, num_total_points, num_contexts, target_y=None):
		"""
		Returns the predicted mean and variance at the target points.

		Args:
			query: Array containing ((context_x, context_y), target_x) where:
			context_x: Array of shape batch_size x num_context x 1 contains the x values of the context points.
			context_y: Array of shape batch_size x num_context x 1 contains the y values of the context points.
			target_x: Array of shape batch_size x num_target x 1 contains the x values of the target points.
			target_y: The ground truth y values of the target y. An array of shape batchsize x num_targets x 1.
			num_total_points: Number of target points.

		Returns:
			log_p: The log_probability of the target_y given the predicted distribution.
			mu: The mean of the predicted distribution.
			sigma: The variance of the predicted distribution.
		"""
		(context_x, context_y), target_x = query

		# Pass query through the encoder and the decoder
		representation = self._encoder(context_x, context_y, num_contexts)
		dist, mu, sigma = self._decoder(representation, target_x, num_total_points)

		# If we want to calculate the log_prob for training we will make use of the
		# target_y. At test time the target_y is not available so we return None
		if target_y is not None:
			log_p = dist.log_prob(target_y)
		else:
			log_p = None

		return log_p, mu, sigma