# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd

from common_util import load_df, 
from recon.common import dum


def test(argv):
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)

	pba_oc_xwhole = load_df('pba_oc_return_fth_of_xwhole.csv', RECON_DIR +'rep' +os.sep +asset +os.sep, data_format='csv')
	pba_oa_xwhole = load_df('pba_oa_return_fth_of_xwhole.csv', RECON_DIR +'rep' +os.sep +asset +os.sep, data_format='csv')
	pba_oc_whole = load_df('pba_oc_return_fth_af_whole.csv', RECON_DIR +'rep' +os.sep +asset +os.sep, data_format='csv')
	pba_oa_whole = load_df('pba_oa_return_fth_af_whole.csv', RECON_DIR +'rep' +os.sep +asset +os.sep, data_format='csv')

	print(pba_oc_xwhole)

	# print(ve)
	
if __name__ == '__main__':
	test(sys.argv[1:])