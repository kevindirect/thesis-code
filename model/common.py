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


# PACKAGE CONSTANTS
DATASET_DIR = RECON_DIR +'dataset' +sep
FILTERSET_DIR = RECON_DIR +'filterset' +sep
KERAS_MODEL_DIR = MODEL_DIR +'model_k' +sep
TENSORFLOW_MODEL_DIR = MODEL_DIR +'model_t' +sep
HOPT_WORKER_BIN = 'hyperopt-mongo-worker'
EXPECTED_NUM_HOURS = 8
TRIALS_COUNT = 100
TEST_RATIO = .2
VAL_RATIO = .25
ERROR_CODE = 999999
OPT_TRANSLATOR = {
	'SGD': SGD,
	'RMSprop': RMSprop,
	'Adam': Adam,
	'Nadam': Nadam
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

