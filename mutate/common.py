#                      __        __
#     ____ ___  __  __/ /_____ _/ /____
#    / __ `__ \/ / / / __/ __ `/ __/ _ \
#   / / / / / / /_/ / /_/ /_/ / /_/  __/
#  /_/ /_/ /_/\__,_/\__/\__,_/\__/\___/
# mutate stage common.
"""
Kevin Patel
"""
# *********** COMMON TO ALL CRUNCH PACKAGES ***********
import sys
from os.path import dirname, realpath

sys.path.insert(0, dirname(dirname(dirname(realpath(sys.argv[0])))))

# ********** SPECIFIC TO THIS CRUNCH PACKAGE **********
# MUTATE
import os
from os import sep
from common_util import MUTATE_DIR, DT_CAL_DAILY_FREQ, load_json

# OTHER STAGE DEPENDENCIES

# PACKAGE CONSTANTS
STANDARD_DAY_LEN = 8 # standard eight hour trading day

# PACKAGE DEFAULTS
dum = 0
GRAPHS_DIR = MUTATE_DIR +'runt-graphs' +sep
HISORY_DIR = MUTATE_DIR + 'runt-history' +sep
TRANSFORMS_DIR = MUTATE_DIR +'runt-transforms' +sep
default_num_sym = 4
default_max_seg = STANDARD_DAY_LEN

# PACKAGE UTIL FUNCTIONS
load_jsons = lambda dir_path: {fname[:-JSON_SFX_LEN]: load_json(fname, dir_path=dir_path) for fname in os.listdir(dir_path)}
get_graphs = partial(load_jsons, GRAPHS_DIR)
get_transforms = partial(load_jsons, TRANSFORMS_DIR)

