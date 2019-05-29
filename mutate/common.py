"""
Kevin Patel
"""
# *********** COMMON TO ALL CRUNCH PACKAGES ***********
import sys
from os.path import dirname, realpath

sys.path.insert(0, dirname(dirname(dirname(realpath(sys.argv[0])))))

# ********** SPECIFIC TO THIS CRUNCH PACKAGE **********
# MUTATE
from os import sep
from common_util import MUTATE_DIR, DT_CAL_DAILY_FREQ

# OTHER STAGE DEPENDENCIES


# PACKAGE CONSTANTS
STANDARD_DAY_LEN = 8 # standard eight hour trading day

# PACKAGE DEFAULTS
dum = 0
GRAPHS_DIR = MUTATE_DIR +'runt-graphs' +sep
TRANSFORMS_DIR = MUTATE_DIR +'runt-transforms' +sep
default_num_sym = 4
default_max_seg = STANDARD_DAY_LEN

# PACKAGE UTIL FUNCTIONS

