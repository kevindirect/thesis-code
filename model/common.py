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
from keras.optimizers import SGD, RMSprop, Adam, Nadam
import torch
from tensorflow.train import RMSPropOptimizer, AdamOptimizer
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits, softmax_cross_entropy_with_logits_v2

# PACKAGE CONSTANTS
DATASET_DIR = RECON_DIR +'dataset' +sep
FILTERSET_DIR = RECON_DIR +'filterset' +sep
HOPT_WORKER_BIN = 'hyperopt-mongo-worker'
ERROR_CODE = 999999
EXPECTED_NUM_HOURS = 8
VAL_RATIO = .2
TEST_RATIO = .2

# Keras
KERAS_MODELS_DIR = MODEL_DIR +'model_k' +sep
KERAS_OPT_TRANSLATOR = {
	'SGD': SGD,
	'RMSprop': RMSprop,
	'Adam': Adam,
	'Nadam': Nadam
}

# PyTorch
PYTORCH_MODELS_DIR = MODEL_DIR +'model_p' +sep
PYTORCH_OPT_TRANSLATOR = {
	'RMSprop': torch.optim.RMSprop,
	'Adam': torch.optim.Adam
}
PYTORCH_LOSS_TRANSLATOR = {
	'bce': torch.nn.BCELoss,						# Binary Probability: Binary Cross Entropy
	'bcel': torch.nn.BCEWithLogitsLoss,				# Binary: Binary Cross Entropy with Logits (Sigmoid + BCELoss)
	'sm': torch.nn.SoftMarginLoss,					# Binary: Soft Margin Loss
	'ce': torch.nn.CrossEntropyLoss,				# Categorical: Cross Entropy Loss (LogSoftmax + NLLLoss)
	'mls': torch.nn.MultiLabelSoftMarginLoss,		# Categorical: Multi Label Soft Margin Loss
	'nll': torch.nn.NLLLoss,						# Categorical: Negative Log Likelihood Loss
	'mae': torch.nn.L1Loss,							# Regression: Mean Absolute Error Loss
	'mse': torch.nn.MSELoss,						# Regression: Mean Squared Error Loss
	'sl1': torch.nn.SmoothL1Loss					# Regression: Smooth L1 Loss
}

# TensorFlow
TENSORFLOW_MODELS_DIR = MODEL_DIR +'model_t' +sep
TENSORFLOW_OPT_TRANSLATOR = {
	'RMSprop': RMSPropOptimizer,
	'Adam': AdamOptimizer
}
TENSORFLOW_LOSS_TRANSLATOR = {
	'sparse_ce': sparse_softmax_cross_entropy_with_logits, 	# This version takes in a 1D series of integer labels
	'onehot_ce': softmax_cross_entropy_with_logits_v2	    # One hot encoded version of sparse_cross_entropy
}
dum = 0


# PACKAGE DEFAULTS
default_ray_config = {
	"init": {
		"num_cpus": 8,
		"num_gpus": 1,
		"redirect_output": True,
		"include_webui": False
	}
}
default_ray_trial_resources = {"cpu": 2, "gpu": 1}
default_model = 'BinTCN'
default_dataset = 'raw_pba_ohlca.json'
default_backend = 'pytorch'
default_trials_count = 100
default_filter = ["0"]
default_nt_filter = ["1"]
default_opt_filter = ["1", "2"]
default_target_col_idx = 0
default_target_idx = [0, 1, 2]


# PACKAGE UTIL FUNCTIONS

