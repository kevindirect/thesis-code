# Kevin Patel

# *********** COMMON TO ALL CRUNCH PACKAGES ***********
import sys
from os.path import dirname, realpath

sys.path.insert(0, dirname(dirname(dirname(realpath(sys.argv[0])))))

# ********** SPECIFIC TO THIS CRUNCH PACKAGE **********
# PACKAGE DEFAULTS
default_joinfile = 'join.json'
default_splitfile = 'split.json'
default_accessfile = 'access.json'

