#                          __     __
#     ____ ___  ____  ____/ /__  / /
#    / __ `__ \/ __ \/ __  / _ \/ /
#   / / / / / / /_/ / /_/ /  __/ /
#  /_/ /_/ /_/\____/\__,_/\___/_/
# model stage common
"""
Kevin Patel
"""

# *********** COMMON TO ALL CRUNCH PACKAGES ***********
import sys
from os.path import dirname, realpath

sys.path.insert(0, dirname(dirname(dirname(realpath(sys.argv[0])))))

# ********** SPECIFIC TO THIS CRUNCH PACKAGE **********
# MODEL
from os import sep
from common_util import RECON_DIR, MODEL_DIR

# OTHER STAGE DEPENDENCIES
import torch
import torch.nn as nn
import torch.optim as optim
"""
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from tensorflow.train import RMSPropOptimizer, AdamOptimizer
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits, softmax_cross_entropy_with_logits_v2
"""

# PACKAGE CONSTANTS
XG_DIR = MODEL_DIR +'xg' +sep
XG_PROCESS_DIR = MODEL_DIR +'xg-process' +sep
XG_DATA_DIR = MODEL_DIR +'xg-data' +sep
XVIZ_DIR = MODEL_DIR +'xg-graphs-viz' +sep
DATASET_DIR = RECON_DIR +'dataset' +sep
FILTERSET_DIR = RECON_DIR +'filterset' +sep
HOPT_WORKER_BIN = 'hyperopt-mongo-worker'
ERROR_CODE = 999999
EXPECTED_NUM_HOURS = 8
INTRADAY_LEN = EXPECTED_NUM_HOURS
VAL_RATIO = .2
TEST_RATIO = .2

# PyTorch
PYTORCH_MODELS_DIR = MODEL_DIR +'model_p' +sep
PYTORCH_ACT_MAPPING = {
	'sig': nn.Sigmoid,
	'tanh': nn.Tanh,
	'smax': nn.Softmax,
	'smax2d': nn.Softmax2d,
	'logsmax': nn.LogSoftmax,
	'splus': nn.Softplus,
	'relu': nn.ReLU,
	'lrelu': nn.LeakyReLU,
	'elu': nn.ELU,
	'celu': nn.CELU,
	'prelu': nn.PReLU,
	'selu': nn.SELU
}
PYTORCH_LOSS_MAPPING = {
	# Binary
	'bce': nn.BCELoss,
	'bcel': nn.BCEWithLogitsLoss,
	'sm': nn.SoftMarginLoss,

	# Categorical
	'ce': nn.CrossEntropyLoss,
	'mls': nn.MultiLabelSoftMarginLoss,
	'nll': nn.NLLLoss,

	# Regression
	'mae': nn.L1Loss,
	'mse': nn.MSELoss,
	'sl1': nn.SmoothL1Loss
}
PYTORCH_OPT_MAPPING = {
	'rmsp': optim.RMSprop,
	'adam': optim.Adam
}
PYTORCH_SCH_MAPPING = {
	'cos': optim.lr_scheduler.CosineAnnealingLR,
	'cyc': optim.lr_scheduler.CyclicLR,
	'exp': optim.lr_scheduler.ExponentialLR,
	'lam': optim.lr_scheduler.LambdaLR,
	'mst': optim.lr_scheduler.MultiStepLR,
	'rpl': optim.lr_scheduler.ReduceLROnPlateau,
	'st': optim.lr_scheduler.StepLR
}

"""
# Keras
KERAS_MODELS_DIR = MODEL_DIR +'model_k' +sep
KERAS_OPT_MAPPING = {
	'SGD': SGD,
	'RMSprop': RMSprop,
	'Adam': Adam,
	'Nadam': Nadam
}
"""

"""
# TensorFlow
TENSORFLOW_MODELS_DIR = MODEL_DIR +'model_t' +sep
TENSORFLOW_OPT_MAPPING = {
	'RMSprop': RMSPropOptimizer,
	'Adam': AdamOptimizer
}
TENSORFLOW_LOSS_MAPPING = {
	'sparse_ce': sparse_softmax_cross_entropy_with_logits, 	# This version takes in a 1D series of integer labels
	'onehot_ce': softmax_cross_entropy_with_logits_v2	    # One hot encoded version of sparse_cross_entropy
}
"""

# PACKAGE DEFAULTS
dum = 0
default_ray_config = {
	"init": {
		"num_cpus": 8,
		"num_gpus": 1,
		"redirect_output": True,
		"include_webui": False
	}
}
default_ray_trial_resources = {"cpu": 2, "gpu": 1}
default_model = 'TCN'
default_xg = 'xg0_reteod_direod.json'
default_dataset = 'xg0_reteod_direod.json'
default_backend = 'pytorch'
default_trials_count = 100
default_filter = ["0"]
default_nt_filter = ["1"]
default_opt_filter = ["1", "2"]
default_target_col_idx = 0
default_target_idx = [0, 1, 2]


# PACKAGE UTIL FUNCTIONS

