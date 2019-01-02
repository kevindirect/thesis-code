"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd
from hyperopt import hp, STATUS_OK
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, AveragePooling1D

from common_util import MODEL_DIR
from model.common import MODELS_DIR, ERROR_CODE
from model.model_k.binary_classifier import BinaryClassifier


class OneLayerBinaryCNN(BinaryClassifier):
	"""One layer binary convolutional neural network classifier."""

	def __init__(self, other_space={}):
		default_space = {
			'layer1_size': hp.choice('layer1_size', [8, 16, 32, 64, 128]),
			'layer1_kernel_size': hp.choice('layer1_kernel_size', [1, 2, 3, 4, 8]),
			'layer1_strides': hp.choice('layer1_strides', [1, 2, 3, 4, 8]),
			'layer1_pooling': hp.choice('layer1_pooling', [None, 'max', 'avg']),
			'layer1_pooling_size': hp.choice('layer1_pooling_size', [1, 2, 3, 4]),
			'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh', 'linear'])
		}
		super(OneLayerBinaryCNN, self).__init__({**default_space, **other_space})

	def make_model(self, params, num_inputs):
		# Define model
		inputs = Input(shape=(num_inputs,), name='inputs')
		layer_one = Conv1D(params['layer1_size'], params['layer1_kernel_size'], strides=params['layer1_strides'], activation=params['activation'])(inputs)

		if (params['layer1_pooling'] is not None):
			if (params['layer1_pooling'] == 'max'):
				layer_one = MaxPooling1D(params['layer1_pooling_size'])(layer_one)
			elif (params['layer1_pooling'] == 'avg'):
				layer_one = AveragePooling1D(params['layer1_pooling_size'])(layer_one)

		output = Dense(1, activation=params['output_activation'], name='output')(layer_one)

		# Compile model
		model = Model(inputs=inputs, outputs=output)
		model.compile(optimizer=self.make_optimizer(params), loss=params['loss'], metrics=self.metrics)

		return model
