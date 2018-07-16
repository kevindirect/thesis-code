# Kevin Patel

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
default_threshfile = 'thresh_all.json'
default_label_threshfile = 'label_thresh_mvp.json'
default_labelfile = 'label_mvp.json'
default_runt_dir_name = 'runt' +sep
default_trfs_dir_name = 'trfs' +sep
default_num_sym = 4
default_max_seg = STANDARD_DAY_LEN

# PACKAGE UTIL FUNCTIONS

