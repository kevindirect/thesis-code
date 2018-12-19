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
from keras.layers import Input, Dense, Dropout

from common_util import MODEL_DIR
from model.common import MODELS_DIR, ERROR_CODE
from model.model.binary_classifier import BinaryClassifier


class ThreeLayerBinaryFFN(BinaryClassifier):
	"""Three layer binary feed forward network classifier."""

	def __init__(self, other_space={}):
		default_space = {
			'layer1_size': hp.choice('layer1_size', [8, 16, 32, 64, 128]),
			'layer1_dropout': hp.uniform('layer1_dropout', .1, .9),
			'layer2_size': hp.choice('layer2_size', [8, 16, 32, 64, 128]),
			'layer2_dropout': hp.uniform('layer2_dropout', .1, .9),
			'layer3_size': hp.choice('layer3_size', [8, 16, 32, 64, 128]),
			'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh', 'linear'])
		}
		super(ThreeLayerBinaryFFN, self).__init__({**default_space, **other_space})

	def make_model(self, params, num_inputs):
		# Define model
		inputs = Input(shape=(num_inputs,), name='inputs')
		layer1 = Dense(params['layer1_size'], activation=params['activation'], kernel_initializer='glorot_uniform', bias_initializer='zeros')(inputs)
		layer1_dropout = Dropout(params['layer1_dropout'])(layer1)
		layer2 = Dense(params['layer2_size'], activation=params['activation'], kernel_initializer='glorot_uniform', bias_initializer='zeros')(layer1_dropout)
		layer2_dropout = Dropout(params['layer2_dropout'])(layer2)
		layer3 = Dense(params['layer3_size'], activation=params['activation'], kernel_initializer='glorot_uniform', bias_initializer='zeros')(layer2_dropout)
		output = Dense(1, activation=params['output_activation'], name='output')(layer3)

		# Compile model
		model = Model(inputs=inputs, outputs=output)
		model.compile(optimizer=params['opt'](lr=params['lr']), loss=params['loss'], metrics=self.metrics)

		return model
