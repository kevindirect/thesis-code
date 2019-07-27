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
from functools import partial
from common_util import MUTATE_DIR, JSON_SFX_LEN, is_valid, load_json

# OTHER STAGE DEPENDENCIES

# PACKAGE CONSTANTS
STANDARD_DAY_LEN = 8 # standard eight hour trading day

# PACKAGE DEFAULTS
dum = 0
GRAPHS_DIR = MUTATE_DIR +'runt-graphs' +sep
HISTORY_DIR = MUTATE_DIR + 'runt-history' +sep
TRANSFORMS_DIR = MUTATE_DIR +'runt-transforms' +sep
RVIZ_DIR = MUTATE_DIR +'runt-viz' +sep
default_num_sym = 4
default_max_seg = STANDARD_DAY_LEN

# PACKAGE UTIL FUNCTIONS
def load_jsons(dir_path, whitelist=None):
	if (is_valid(whitelist)):
		in_wl = lambda f: any(f.startswith(w) for w in whitelist)
		d = {fname[:-JSON_SFX_LEN]: load_json(fname, dir_path=dir_path) for fname in os.listdir(dir_path) if (in_wl(fname))}
	else:
		d = {fname[:-JSON_SFX_LEN]: load_json(fname, dir_path=dir_path) for fname in os.listdir(dir_path)}
	return d

get_graphs = partial(load_jsons, dir_path=GRAPHS_DIR)
get_transforms = partial(load_jsons, dir_path=TRANSFORMS_DIR)

