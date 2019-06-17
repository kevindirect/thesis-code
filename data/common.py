#         __      __
#    ____/ /___ _/ /_____ _
#   / __  / __ `/ __/ __ `/
#  / /_/ / /_/ / /_/ /_/ /
#  \__,_/\__,_/\__/\__,_/
# data stage common.
"""
Kevin Patel
"""
# *********** COMMON TO ALL CRUNCH PACKAGES ***********
import sys
from os.path import dirname, realpath

sys.path.insert(0, dirname(dirname(dirname(realpath(sys.argv[0])))))

# ********** SPECIFIC TO THIS CRUNCH PACKAGE **********
from os import sep
from common_util import DATA_DIR

# PACKAGE CONSTANTS
dum=0
DR_FNAME = 'data_record'
DR_FMT = 'csv'

# Data Record Fields
DR_IDS = ['id', 'name']						# Autogen numerical indices
DR_CAT = ['cat']						# Data category
DR_DIR = ['root', 'basis', 'stage', 'type', 'freq']		# Info fields used to make path
DR_INFO = ['desc', 'hist']					# Other info
DR_GEN = ['dir', 'size', 'dumptime', 'hash', 'created', 'modified']	# Other autogen columns

# Useful field lists
DR_NAME = ['root', 'stage', 'id']		# Name field components
DR_MAN = DR_CAT + DR_DIR + DR_INFO		# Manually entered (required) fields
DR_COLS = DR_IDS + DR_MAN + DR_GEN		# All fields

DG_PFX = 'dg_'
CS_PFX = 'cs_'

AXEFILES_DIR = DATA_DIR +'axefiles' +sep

# PACKAGE DEFAULTS
default_cleanfile = 'clean.json'
default_viewfile_sfx = 'view'

