"""
Kevin Patel
"""
import sys
import os
from os.path import sep, basename, dirname
import logging

import numpy as np
import pandas as pd
import optuna

from common_util import MODEL_DIR, dump_df, benchmark, get_cmd_args
from model.common import ASSETS, INTERVAL_YEARS, OPTUNA_DB_FNAME, OPTUNA_CSV_FNAME


def optuna_view(argv):
	cmd_arg_list = ['dump-csv', 'model=', 'assets=', 'xdata=', 'ydata=', 'optuna-monitor=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__), \
		script_pkg=basename(dirname(__file__)))
	dump_csv = cmd_input['dump-csv']
	model_name = cmd_input['model='] or 'anp'
	asset_name = cmd_input['assets='] or ASSETS[0]
	fdata_name = cmd_input['xdata='] or 'h_pba_h'
	ldata_name = cmd_input['ydata='] or 'ddir'
	monitor = cmd_input['optuna-monitor='] or 'val_loss'
	optimize_min = {
		'val_loss': True
	}.get(monitor, False)
	interval = INTERVAL_YEARS
	data_name = (f'{interval[0]}_{interval[1]}'
		f'_{ldata_name}_{fdata_name}').replace(',', '_')

	# model options: stcn, anp
	if (model_name in ('stcn', 'StackedTCN', 'GenericModel_StackedTCN')):
		from model.pl_generic import GenericModel
		from model.model_util import StackedTCN
		pl_model_fn, pt_model_fn = GenericModel, StackedTCN
	elif (model_name in ('anp', 'AttentiveNP', 'NPModel_AttentiveNP')):
		from model.pl_np import NPModel
		from model.np_util2 import AttentiveNP
		pl_model_fn, pt_model_fn = NPModel, AttentiveNP
	model_name = f'{pl_model_fn.__name__}_{pt_model_fn.__name__}'

	study_dir = MODEL_DIR \
		+sep.join(['olog', model_name, asset_name, data_name, monitor]) +sep
	study_name = ','.join([model_name, asset_name, data_name, monitor])
	study_db_path = f'sqlite:///{study_dir}{OPTUNA_DB_FNAME}'

	print(f'study name:  {study_name}')
	print(f'study dir:   {study_dir}')
	print(f'study db:    {study_db_path}')
	print()

	study_df = optuna.load_study(storage=study_db_path, study_name=study_name) \
		.trials_dataframe().sort_values(by='value')
	if (dump_csv):
		dump_df(study_df.set_index('number'), OPTUNA_CSV_FNAME, dir_path=study_dir, \
			data_format='csv')
	print(df_study_stats(study_df, optimize_min))

def df_study_stats(study_df: pd.DataFrame, optimize_min: bool):
	"""
	"""
	completed_trials = study_df.loc[study_df['state'] == 'COMPLETE']
	pruned_trials = study_df.loc[study_df['state'] == 'PRUNED']
	top5 = completed_trials.nsmallest(5, 'value', keep='all')
	bot5 = completed_trials.nlargest(5, 'value', keep='all')
	shortest = completed_trials.nsmallest(1, 'duration', keep='all') \
		[['number', 'duration']]
	longest = completed_trials.nlargest(1, 'duration', keep='all') \
		[['number', 'duration']]

	overall = (
		f'total trials:            {len(study_df)}\n'
		f'completed trials:        {len(completed_trials)}\n'
		f'pruned trials:           {len(pruned_trials)}\n'
		f'running or null trials:  {len(study_df)-(len(completed_trials)+len(pruned_trials))}\n\n'
	)

	low_stat = (
		f'lowest 5:\n{top5}\n'
		f'lowest params:\n{top5.iloc[0]}\n\n'
	)

	high_stat = (
		f'highest 5:\n{bot5}\n'
		f'highest params:\n{bot5.iloc[0]}\n\n'
	)

	time_stat = (
		f'shortest completed:   {shortest.duration.iloc[0]} ({shortest.number.iloc[0]})\n'
		f'longest completed:    {longest.duration.iloc[0]} ({longest.number.iloc[0]})\n'
		f'avg trial completion: {completed_trials.duration.mean(axis=0)}\n'
		f'total study duration: {completed_trials.duration.sum(axis=0)}\n'
	)

	return overall + (low_stat+high_stat if (optimize_min) else high_stat+low_stat) + time_stat

if __name__ == '__main__':
	optuna_view(sys.argv[1:])

