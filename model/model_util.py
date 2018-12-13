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
from model.model.OneBinCNN import OneLayerBinaryCNN
from model.model.OneBinGRU import OneLayerBinaryGRU
from model.model.OneBinLCL import OneLayerBinaryLCL
from model.model.OneBinLSTM import OneLayerBinaryLSTM
from model.model.ThreeBinFFN import ThreeLayerBinaryFFN


""" ********** MODELS ********** """
BINARY_CLF_MAP = {
	'OneBinCNN': OneLayerBinaryCNN,
	'OneBinGRU': OneLayerBinaryGRU,
	'OneBinLCL': OneLayerBinaryLCL,
	'OneBinLSTM': OneLayerBinaryLSTM,
	'ThreeBinFFN': ThreeLayerBinaryFFN
}
