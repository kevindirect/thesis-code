# Kevin Patel

import sys
import os
import logging
import numpy as np
import pandas as pd
import engarde.decorators as ed

from common_util import load_df, dump_df, DATA_DIR
from data.data_api import DataAPI


def getTEvents(gRaw, h):
	"""
	Symmetric CUSUM Filter where Eₜ₋₁[yₜ] = yₜ₋₁
	Lopez De Prado, Advances in Financial Machine Learning (p. 39)
	"""
	tEvents, sPos, sNeg = [], 0, 0
	diff = gRaw.diff()

	for i in diff.index[1:]:
		sPos, sNeg = max(0, sPos+diff.loc[i]), min(0, sNeg+diff.loc[i])

		if (sNeg < -h):
			sNeg = 0
			tEvents.append(i)
		elif (sPos > h):
			sPos = 0
			tEvents.append(i)

	return pd.DatetimeIndex(tEvents)


