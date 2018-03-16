# Kevin Patel

import numpy as np
import pandas as pd
import sys
import os
from common_util import DATA_DIR
from eda.common import dummy
from data.data_api import DataAPI


def main(argv):

	search_terms = {
		'stage': 'raw'
	}
	for rec, df in DataAPI.generate(search_terms):
		df.corr()



if __name__ == '__main__':
	main(sys.argv[1:])
