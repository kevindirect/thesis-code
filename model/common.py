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
from tensorflow.train import RMSPropOptimizer, AdamOptimizer
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits, softmax_cross_entropy_with_logits_v2

# PACKAGE CONSTANTS
DATASET_DIR = RECON_DIR +'dataset' +sep
FILTERSET_DIR = RECON_DIR +'filterset' +sep
HOPT_WORKER_BIN = 'hyperopt-mongo-worker'
ERROR_CODE = 999999
TRIALS_COUNT = 100
EXPECTED_NUM_HOURS = 8
VAL_RATIO = .2
TEST_RATIO = .2

# Keras
KERAS_MODEL_DIR = MODEL_DIR +'model_k' +sep
KERAS_OPT_TRANSLATOR = {
	'SGD': SGD,
	'RMSprop': RMSprop,
	'Adam': Adam,
	'Nadam': Nadam
}

# TensorFlow
TENSORFLOW_MODEL_DIR = MODEL_DIR +'model_t' +sep
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
default_model = 'OneBinGRU'
default_dataset = 'mvp_dnorm_raw_pba_avgprice.json'
default_filter = ["0"]
default_nt_filter = ["1"]
default_opt_filter = ["1", "2"]
default_target_col_idx = 0
default_target_idx = [0, 1, 2]


# PACKAGE UTIL FUNCTIONS

