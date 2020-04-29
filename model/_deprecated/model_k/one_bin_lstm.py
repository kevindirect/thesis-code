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
from keras.layers import Input, Dense, LSTM

from common_util import MODEL_DIR
from model.common import KERAS_MODELS_DIR, ERROR_CODE
from model.model_k.sequential_mixin import SequentialMixin
from model.model_k.binary_classifier import BinaryClassifier


class OneLayerBinaryLSTM(SequentialMixin, BinaryClassifier):
	"""One layer binary long short term memory classifier."""

	def __init__(self, other_space={}):
		default_space = {
			'step_size': hp.choice('step_size', [3, 5]), 	# AKA target delay or time delay in RNN research
			'layer1_size': hp.choice('layer1_size', [8, 16, 32, 64, 128]),
			'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh', 'linear']),
			'recurrent_activation': hp.choice('recurrent_activation', ['hard_sigmoid']),
			'stateful': hp.choice('stateful', [False])
		}
		super(OneLayerBinaryLSTM, self).__init__({**default_space, **other_space})

	def make_model(self, params, num_inputs):
		# Define model
		inputs = Input(shape=(params['step_size'], num_inputs), name='inputs')
		layer_one = LSTM(params['layer1_size'], activation=params['activation'], return_sequences=False, return_state=False,
			recurrent_activation=params['recurrent_activation'], stateful=params['stateful'])(inputs)
		output = Dense(1, activation=params['output_activation'], name='output')(layer_one)

		# Compile model
		model = Model(inputs=inputs, outputs=output)
		model.compile(optimizer=self.make_optimizer(params), loss=params['loss'], metrics=self.metrics)

		return model
