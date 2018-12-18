# Kevin Patel

# *********** COMMON TO ALL CRUNCH PACKAGES ***********
import sys
from os.path import dirname, realpath

sys.path.insert(0, dirname(dirname(dirname(realpath(sys.argv[0])))))

# ********** SPECIFIC TO THIS CRUNCH PACKAGE **********
# REPORT
from os import sep
from common_util import REPORT_DIR

# OTHER STAGE DEPENDENCIES


# PACKAGE CONSTANTS
MONGOD_BIN = "mongod"
MONGOD_PORT = 27017
MONGOD_ARGS = [
	"--quiet"
]
MONGOD_CHECK_ATTEMPTS = 200
DB_DIR = REPORT_DIR +"db" +sep


# PACKAGE DEFAULTS

# PACKAGE UTIL FUNCTIONS

