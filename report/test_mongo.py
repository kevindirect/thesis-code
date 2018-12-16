"""
Kevin Patel
"""
import sys
import os
from os.path import basename
import ssl
import logging

import numpy as np
import pandas as pd
import pymongo

from common_util import REPORT_DIR, JSON_SFX_LEN, DT_CAL_DAILY_FREQ, load_json, str_to_list, get_cmd_args, in_debug_mode, pd_common_index_rows, benchmark
from model.common import remote_file


def test_mongo(argv):
	cmd_arg_list = []
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__))

	rem = load_json(remote_file, dir_path=REPORT_DIR)
	cnn = rem["uri"].format(**rem["cred"])
	myclient = pymongo.MongoClient(cnn, ssl=True, ssl_cert_reqs=ssl.CERT_REQUIRED, ssl_ca_certs=rem["cred"]["cert"])

	mydb = myclient["mydatabase"]
	print(myclient.list_database_names())

	dblist = myclient.list_database_names()
	if "mydatabase" in dblist:
		print("The database exists.")
	


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		test_mongo(sys.argv[1:])
