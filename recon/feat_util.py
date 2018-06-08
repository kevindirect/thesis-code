# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common_util import remove_dups_list
from recon.common import dum


def split_ser(ser, num_cols, pfx=''):
	split_df = pd.DataFrame(index=ser.index)
	column_names = [str(pfx +'_' +str(i)) for i in range(num_cols)]
	# splitted = ser.str.split(',', num_cols, expand=True)
	split_df[column_names] = ser.str.split(',', num_cols, expand=True)
	# split_df = pd.DataFrame(data=splitted, index=ser.index, columns=column_names)
	return split_df
