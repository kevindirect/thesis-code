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


# PACKAGE CONSTANTS
DATASET_DIR = RECON_DIR +'dataset' +sep
FILTERSET_DIR = RECON_DIR +'filterset' +sep
MODELS_DIR = MODEL_DIR +'model' +sep
HOPT_WORKER_BIN = 'hyperopt-mongo-worker'
EXPECTED_NUM_HOURS = 8
TRIALS_COUNT = 100
TEST_RATIO = .2
VAL_RATIO = .25
ERROR_CODE = 999999
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
default_dataset = 'mvp_dnorm_raw.json'
default_filter = ["0"]
default_nt_filter = ["1"]
default_opt_filter = ["1", "2"]
default_target_col_idx = 0
default_target_idx = [0, 1, 2]


# PACKAGE UTIL FUNCTIONS

