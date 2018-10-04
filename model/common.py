# Kevin Patel

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
EXPECTED_NUM_HOURS = 8
dum = 0


# PACKAGE DEFAULTS
default_dataset = 'dnorm_sym.json'
default_filterset = 'default_dnorm_sym.json'
default_filter = ["0"]
default_nt_filter = ["1"]
default_target_col_idx = 0

# PACKAGE UTIL FUNCTIONS

