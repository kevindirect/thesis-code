"""
Kevin Patel

Adapted from code by Weiping Song
Source: https://github.com/Songweiping/TCN-TF

Author: Weiping Song
Time: April 24, 2018
"""
import sys
import os
from os import sep
import logging

import numpy as np
import pandas as pd
from hyperopt import hp, STATUS_OK, STATUS_FAIL
import tensorflow as tf

from common_util import MODEL_DIR
from model.common import TENSORFLOW_MODEL_DIR
from model.model_t.tcn_util import TemporalConvNet
from model.model_t.binary_classifier import BinaryClassifier


class TCN(BinaryClassifier):
	"""
	Temporal convolutional neural network classifier.
	Adds a dense layer to convert the embedding into a classification.
	"""
	def __init__(self, other_space={}):
		default_space = {
			# 'input_mult': hp.choice('input_mult', [1, 2, 4, 8]),
			'topology': hp.choice('topology', [])
			'num_channels': hp.choice('num_channels', [.2]),
			'stride': hp.choice('stride', [1]),
			'kernel_size': hp.choice('kernel_size', [2]),
			'dropout': hp.choice('dropout', [.2]),
			'num_channels': hp.choice('num_channels', [.2]),
		}
		super(TCN, self).__init__({**default_space, **other_space})

	def preproc(self, params, data):
		"""
		Apply any final transforms or reshaping to passed data tuple before fitting.
		"""
		return identity_fn(data)

	def make_model(self, params, num_inputs):
		x = tf.placeholder(tf.int32, shape=(num_inputs, None))
		y = tf.placeholder(tf.int32, shape=(None, None))
		self.eff_history = tf.placeholder(tf.int32, shape=None, name='eff_history')
		self.dropout = tf.placeholder_with_default(0., shape=())
		# self.emb_dropout = tf.placeholder_with_default(0., shape=())

		embedding = tf.get_variable('char_embedding', shape=(self.output_size, self.emb_size), dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1))
		inputs = tf.nn.embedding_lookup(embedding, self.x)

		self.tcn = TemporalConvNet(self.num_channels, stride=1, kernel_size=self.kernel_size, dropout=self.dropout)
		outputs = self.tcn(inputs)
		# reshaped_outputs = tf.reshape(outputs, (-1, self.emb_size))
		# logits = tf.matmul(reshaped_outputs, embedding, transpose_b=True)

		logits_shape = tf.concat([tf.shape(outputs)[:2], (tf.constant(self.output_size),)], 0)
		logits = tf.reshape(logits, shape=logits_shape)
		eff_logits = tf.slice(logits, [0,self.eff_history,0], [-1, -1, -1])
		eff_labels = tf.slice(self.y, [0,self.eff_history], [-1, -1])
		CE_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=eff_labels, logits=eff_logits)
		self.loss = tf.reduce_mean(CE_loss)

		logits_shape = tf.concat([tf.shape(outputs)[:2], (tf.constant(self.output_size),)], 0)
		mdl_logits = tf.reshape(logits, shape=logits_shape)
		loss_fn = TENSORFLOW_LOSS_TRANSLATOR.get(params['loss'])
		ce_loss = loss_fn(labels=eff_labels, logits=mdl_logits)

		optimizer = make_optimizer(params)
		optimizer.minimize(ce_loss)

		# gvs = optimizer.compute_gradients(self.loss)
		# capped_gvs = [(tf.clip_by_value(grad, -self.clip_value, self.clip_value), var) for grad, var in gvs]
		# self.train_op = optimizer.apply_gradients(capped_gvs)

	# def make_model_fn(params):
	# 	"""
	# 	session_config = tf.ConfigProto()
	# 	session_config.gpu_options.allow_growth = True
	# 	estimator_config = tf.estimator.RunConfig(session_config=session_config)
	# 	my_estimator = tf.estimator.Estimator(..., config=estimator_config)
	# 	"""

	# 	def model_fn(features, labels, mode):
	# 		# Build the neural network
	# 		# Because Dropout have different behavior at training and prediction time, we
	# 		# need to create 2 distinct computation graphs that still share the same weights.
	# 		logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)
	# 		logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)
			
	# 		# Predictions
	# 		pred_classes = tf.argmax(logits_test, axis=1)
	# 		pred_probas = tf.nn.softmax(logits_test)
			
	# 		# If prediction mode, early return
	# 		if mode == tf.estimator.ModeKeys.PREDICT:
	# 			return tf.estimator.EstimatorSpec(mode, predictions=pred_classes) 

	# 		# Define loss and optimizer
	# 		loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
	# 		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	# 		train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
			
	# 		# Evaluate the accuracy of the model
	# 		acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
			
	# 		# TF Estimators requires to return a EstimatorSpec, that specify
	# 		# the different ops for training, evaluating, ...
	# 		estim_specs = tf.estimator.EstimatorSpec(
	# 			mode=mode,
	# 			predictions=pred_classes,
	# 			loss=loss_op,
	# 			train_op=train_op,
	# 			eval_metric_ops={'accuracy': acc_op})

	# 		return estim_specs

	# 	return model_fn

	def __init__(self, input_size, output_size, num_channels, seq_len, emb_size, kernel_size=2, clip_value=1.0):
		self.input_size = input_size
		self.output_size = output_size
		self.num_channels = num_channels
		self.seq_len = seq_len
		self.kernel_size = kernel_size
		self.emb_size = emb_size
		self.clip_value = clip_value

		self._build()
		self.saver = tf.train.Saver()

	def _build(self):
		self.x = tf.placeholder(tf.int32, shape=(None, None), name='input_chars')
		self.y = tf.placeholder(tf.int32, shape=(None, None), name='next_chars')
		self.lr = tf.placeholder(tf.float32, shape=None, name='lr')
		self.eff_history = tf.placeholder(tf.int32, shape=None, name='eff_history')
		self.dropout = tf.placeholder_with_default(0., shape=())
		self.emb_dropout = tf.placeholder_with_default(0., shape=())

		embedding = tf.get_variable('char_embedding', shape=(self.output_size, self.emb_size), dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1))
		inputs = tf.nn.embedding_lookup(embedding, self.x)

		self.tcn = TemporalConvNet(self.num_channels, stride=1, kernel_size=self.kernel_size, dropout=self.dropout)
		outputs = self.tcn(inputs)
		reshaped_outputs = tf.reshape(outputs, (-1, self.emb_size))
		logits = tf.matmul(reshaped_outputs, embedding, transpose_b=True)

		logits_shape = tf.concat([tf.shape(outputs)[:2], (tf.constant(self.output_size),)], 0)
		logits = tf.reshape(logits, shape=logits_shape)
		eff_logits = tf.slice(logits, [0,self.eff_history,0], [-1, -1, -1])
		eff_labels = tf.slice(self.y, [0,self.eff_history], [-1, -1])
		CE_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=eff_logits, labels=eff_labels)
		self.loss = tf.reduce_mean(CE_loss)

		optimizer = tf.train.GradientDescentOptimizer(self.lr)
		gvs = optimizer.compute_gradients(self.loss)
		capped_gvs = [(tf.clip_by_value(grad, -self.clip_value, self.clip_value), var) for grad, var in gvs]
		self.train_op = optimizer.apply_gradients(capped_gvs)
