"""
Kevin Patel
"""
import sys
import os
import logging

import numpy as np
import pandas as pd

from common_util import identity_fn
from model.common import EXPECTED_NUM_HOURS
from model.model_k.one_bin_cnn import OneLayerBinaryCNN
from model.model_k.one_bin_gru import OneLayerBinaryGRU
from model.model_k.one_bin_lcl import OneLayerBinaryLCL
from model.model_k.one_bin_lstm import OneLayerBinaryLSTM
from model.model_k.three_bin_ffn import ThreeLayerBinaryFFN
from model.model_p.cnn import CNN_CLF
from model.model_p.tcn import TCN_CLF, TCN_REG


""" ********** KERAS BINARY CLASSIFIERS ********** """
KERAS_BINARY_CLF_MAP = { # TODO - fix
	'OneBinCNN': OneLayerBinaryCNN,
	'OneBinGRU': OneLayerBinaryGRU,
	'OneBinLCL': OneLayerBinaryLCL,
	'OneBinLSTM': OneLayerBinaryLSTM,
	'ThreeBinFFN': ThreeLayerBinaryFFN
}

""" ********** PYTORCH CLASSIFIERS ********** """
PYTORCH_CLF_MAP = {
	'CNN': CNN_CLF,
	'TCN': TCN_CLF
}

""" ********** ALL CLASSIFIERS ********** """
CLF_MAP = {
	'keras': KERAS_BINARY_CLF_MAP,
	'pytorch': PYTORCH_CLF_MAP
}

""" ********** PYTORCH REGRESSORS ********** """
PYTORCH_REG_MAP = {
	'TCN': TCN_REG
}

""" ********** ALL REGRESSORS ********** """
REG_MAP = {
	'pytorch': PYTORCH_REG_MAP
}