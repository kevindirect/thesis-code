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
from functools import partial
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
FR_DIR = MODEL_DIR +'fr' +sep
XG_DIR = MODEL_DIR +'xg' +sep
XG_PROCESS_DIR = MODEL_DIR +'xg-process' +sep
XG_DATA_DIR = MODEL_DIR +'xg-data' +sep
XG_VIZ_DIR = MODEL_DIR +'xg-graphs-viz' +sep
DATASET_DIR = RECON_DIR +'dataset' +sep
FILTERSET_DIR = RECON_DIR +'filterset' +sep
XG_INDEX_FNAME = '.index.json'
EXPECTED_NUM_HOURS = 8
INTRADAY_LEN = 8
ASSETS = ('sp_500', 'russell_2000', 'nasdaq_100', 'dow_jones')
INTERVAL_YEARS = (2007, 2018)
WIN_SIZE = 20
VAL_RATIO, TEST_RATIO = .2, .2
# HOPT_WORKER_BIN = 'hyperopt-mongo-worker'
# ERROR_CODE = 999999

# PyTorch
PYTORCH_MODELS_DIR = MODEL_DIR +'model_p' +sep
PYTORCH_ACT1D_LIST = ('relu', 'lrelu', 'elu', 'celu', 'gelu', 'prelu', \
	'selu', 'sig', 'tanh', 'splus', 'smax', 'logsmax')
PYTORCH_ACT_MAPPING = {
	'relu': nn.ReLU,
	'lrelu': nn.LeakyReLU,
	'elu': nn.ELU,
	'celu': nn.CELU,
	'gelu': nn.GELU,
	'prelu': nn.PReLU,
	'selu': nn.SELU,
	'sig': nn.Sigmoid,
	'tanh': nn.Tanh,
	'splus': nn.Softplus,
	'smax': partial(nn.Softmax, dim=0),
	'logsmax': partial(nn.LogSoftmax, dim=0),
	'smax2d': nn.Softmax2d
}
PYTORCH_INIT_LIST = ('zeros', 'ones', 'normal', 'orthogonal', \
	'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal')
PYTORCH_LOSS_MAPPING = {
	# Binary
	'bce': nn.BCELoss,
	'bcel': nn.BCEWithLogitsLoss,
	'sm': nn.SoftMarginLoss,

	# Categorical
	'ce': nn.CrossEntropyLoss,
	'nll': nn.NLLLoss,
	'ml': nn.MultiLabelMarginLoss,
	'mls': nn.MultiLabelSoftMarginLoss,

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

# Optuna
OPTUNA_DB_FNAME = 'trials.db'
OPTUNA_CSV_FNAME = 'trials.csv'
OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT_HOURS = 12

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
dum = None
# default_ray_config = {
# 	"init": {
# 		"num_cpus": 8,
# 		"num_gpus": 1,
# 		"redirect_output": True,
# 		"include_webui": False
# 	}
# }
# default_ray_trial_resources = {"cpu": 2, "gpu": 1}
# default_model = 'TCN'
# default_xg = 'xg0_reteod_direod.json'
# default_dataset = 'xg0_reteod_direod.json'
# default_backend = 'pytorch'
# default_trials_count = 100
# default_filter = ["0"]
# default_nt_filter = ["1"]
# default_opt_filter = ["1", "2"]
# default_target_col_idx = 0
# default_target_idx = [0, 1, 2]


# PACKAGE UTIL FUNCTIONS

