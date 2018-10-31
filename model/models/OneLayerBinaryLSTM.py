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
from keras.layers import Input, Dense, Activation, Dropout, LSTM, GRU
from keras.layers import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, RepeatVector

from common_util import MODEL_DIR
from model.common import MODELS_DIR, ERROR_CODE
from model.models.BinaryClassifierExperiment import BinaryClassifierExperiment


class OneLayerBinaryLSTM(BinaryClassifierExperiment):
	"""One layer binary LSTM classifier."""

	def __init__(self, other_space={}):
		default_space = {
			'layer1_size': hp.choice('layer1_size', [8, 16, 32, 64, 128]),
			'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh', 'linear'])
		}
		super(OneLayerBinaryLSTM, self).__init__({**default_space, **other_space})

	def make_model(self, params, input_shape):
		main_input = Input(shape=input_shape, name='main_input')
		x = LSTM(params['layer1_size'], activation=params['activation'])(main_input)
		output = Dense(1, activation = params['output_activation'], name='output')(x)
		final_model = Model(inputs=[main_input], outputs=[output])
		opt = params['opt'](lr=params['lr'])
		model = final_model.compile(optimizer=opt, loss=params['loss'])

		return model
