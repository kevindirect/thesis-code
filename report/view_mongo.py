"""
Kevin Patel
"""
import sys
import os
from os.path import basename, expanduser
import ssl
import logging

import numpy as np
import pandas as pd
import pymongo

from common_util import REPORT_DIR, JSON_SFX_LEN, DT_CAL_DAILY_FREQ, load_json, str_to_list, get_cmd_args, in_debug_mode, pd_common_index_rows, benchmark
from report.common import DB_DIR, DEFAULT_MONGO_DBS
from report.mongo_server import MongoServer


def test_mongo(argv):
	cmd_arg_list = []
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__))
	with MongoServer() as mongodb:
		client = mongodb.get_client()
		dblist = client.list_database_names()
		print('dbs: {}'.format(dblist))

		for db_name in filter(lambda dbn: dbn not in DEFAULT_MONGO_DBS, dblist):
			print(db_name)
			print(client[db_name].list_collection_names())

	try:
		client.server_info()
	except:
		print('Mongo instance is stopped')


if __name__ == '__main__':
	with benchmark('time to finish') as b:
		test_mongo(sys.argv[1:])
