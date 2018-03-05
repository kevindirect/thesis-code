# Kevin Patel

# *********** COMMON TO ALL CRUNCH PACKAGES ***********
import sys
from os.path import dirname, realpath

sys.path.insert(0, dirname(dirname(dirname(realpath(sys.argv[0])))))

# ********** SPECIFIC TO THIS CRUNCH PACKAGE **********
# RAW

from os.path import basename, isfile
import pandas as pd
from common_util import CRUNCH_DIR, RAW_DIR

# PACKAGE CONSTANTS
TRMI_CONFIG_FNAME = 'trmi.json'
TRMI_CONFIG_DIR = CRUNCH_DIR

# PACKAGE DEFAULTS
default_pricefile = 'richard@marketpsychdata.com--N166567660.csv'
default_pathsfile = 'paths.json'
default_columnsfile = 'columns.json'
default_rowsfile = 'rows.json'

# PACKAGE UTIL FUNCTIONS
def load_csv_no_idx(fname, dir_path=None, local_csv=True):
	fpath = str(dir_path + fname) if dir_path else fname

	if (local_csv and not isfile(fpath)):
		print(basename(fpath), 'must be in:', dirname(fpath))
		sys.exit(2)
	
	return pd.read_csv(fpath)
