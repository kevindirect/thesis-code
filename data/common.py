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

# PACKAGE DEFAULTS

