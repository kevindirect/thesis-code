# Kevin Patel

# *********** COMMON TO ALL CRUNCH PACKAGES ***********
import sys
from os.path import dirname, realpath

sys.path.insert(0, dirname(dirname(dirname(realpath(sys.argv[0])))))

# ********** SPECIFIC TO THIS CRUNCH PACKAGE **********
# MUTATE
import pandas as pd

# OTHER STAGE DEPENDENCIES


# PACKAGE CONSTANTS


# PACKAGE DEFAULTS
dum = 0
default_threshfile = 'thresh.json'
default_labelfile = 'label.json'
default_pattern_threshfile = 'pattern_thresh.json'


# PACKAGE UTIL FUNCTIONS
count_nonnan = lambda df: len(df) - df.isnull().sum()
count_nonzero = lambda df: df.apply(lambda ser: (ser.dropna(axis=0, how='any')!=0).sum())
count_both = lambda df: pd.concat([count_nonnan(df), count_nonzero(df)], axis=1, names=['non_nan', 'non_zero'])

