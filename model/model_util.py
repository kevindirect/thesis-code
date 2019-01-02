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


""" ********** MODELS ********** """
KERAS_BINARY_CLF_MAP = {
	'OneBinCNN': OneLayerBinaryCNN,
	'OneBinGRU': OneLayerBinaryGRU,
	'OneBinLCL': OneLayerBinaryLCL,
	'OneBinLSTM': OneLayerBinaryLSTM,
	'ThreeBinFFN': ThreeLayerBinaryFFN
}
