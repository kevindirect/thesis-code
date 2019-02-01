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
from model.model_p.bin_cnn import BinaryCNN
from model.model_p.bin_tcn import BinaryTCN


""" ********** KERAS BINARY CLASSIFIERS ********** """
KERAS_BINARY_CLF_MAP = {
	'OneBinCNN': OneLayerBinaryCNN,
	'OneBinGRU': OneLayerBinaryGRU,
	'OneBinLCL': OneLayerBinaryLCL,
	'OneBinLSTM': OneLayerBinaryLSTM,
	'ThreeBinFFN': ThreeLayerBinaryFFN
}

""" ********** PYTORCH BINARY CLASSIFIERS ********** """
PYTORCH_BINARY_CLF_MAP = {
	'BinCNN': BinaryCNN,
	'BinTCN': BinaryTCN
}

""" ********** ALL BINARY CLASSIFIERS ********** """
BINARY_CLF_MAP = {
	'keras': KERAS_BINARY_CLF_MAP,
	'pytorch': PYTORCH_BINARY_CLF_MAP
}