# Kevin Patel

# *********** COMMON TO ALL CRUNCH PACKAGES ***********
import sys
from os.path import dirname, realpath

sys.path.insert(0, dirname(dirname(dirname(realpath(sys.argv[0])))))

# ********** SPECIFIC TO THIS CRUNCH PACKAGE **********
# RECON
import pandas as pd
from os import sep

# OTHER STAGE DEPENDENCIES


# PACKAGE CONSTANTS
dum = 0


# PACKAGE DEFAULTS
default_pipefile = 'pipe.json'
default_cv_file = 'cv_kfold.json'


# PACKAGE UTIL FUNCTIONS

