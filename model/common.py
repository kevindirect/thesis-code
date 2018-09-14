# Kevin Patel

# *********** COMMON TO ALL CRUNCH PACKAGES ***********
import sys
from os.path import dirname, realpath

sys.path.insert(0, dirname(dirname(dirname(realpath(sys.argv[0])))))

# ********** SPECIFIC TO THIS CRUNCH PACKAGE **********
# MODEL
from os import sep
from common_util import MODEL_DIR

# OTHER STAGE DEPENDENCIES


# PACKAGE CONSTANTS
DATASET_DIR = MODEL_DIR +'dataset' +sep
dum = 0


# PACKAGE DEFAULTS
default_dataset = 'default.json'


# PACKAGE UTIL FUNCTIONS

