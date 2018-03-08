# Kevin Patel

# *********** COMMON TO ALL CRUNCH PACKAGES ***********
import sys
from os.path import dirname, realpath

sys.path.insert(0, dirname(dirname(dirname(realpath(sys.argv[0])))))

# ********** SPECIFIC TO THIS CRUNCH PACKAGE **********
# DATA

# PACKAGE CONSTANTS
DR_NAME = 'data_record'
DR_FMT = 'csv'

NAME_IDX = 2
DIR_IDX = 3

DR_IDS = ['id', 'name', 'dir']										# Autogenerated columns
DR_REQ = ['root', 'basis', 'stage']									# Minimum needed to dump a df
DR_STAGE = ['transform', 'eda_group']								# Stage specific dump requirements, prepended by stage name
DR_META = ['raw_cat', 'history']									# Other misc metadata (mutable: all)
DR_GEN = ['size', 'dumptime', 'hash', 'created', 'modified'] 		# Other autogenerated columns (mutable: all)
DR_COLS = DR_IDS + DR_REQ + DR_STAGE + DR_META + DR_GEN
