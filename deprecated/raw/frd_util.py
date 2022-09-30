"""
Kevin Patel
"""
import sys
from os import sep
from os.path import basename
import logging

from common_util import RAW_DIR, load_json, get_cmd_args, is_valid, isnt, makedir_if_not_exists, dump_df
from raw.common import default_pricefile, default_pathsfile, default_columnsfile, default_rowsfile, load_csv_no_idx




# if __name__ == '__main__':
# 	(sys.argv[1:])
