"""
Kevin Patel
"""
import sys
import os
from os import sep
from os.path import basename
import logging

import ray
from ray.tune import Experiment, run_experiments, register_trainable
from ray.tune.schedulers import HyperBandScheduler, AsyncHyperBandScheduler
from ray.tune.suggest import HyperOptSearch

from common_util import MODEL_DIR, REPORT_DIR, JSON_SFX_LEN, get_class_name, makedir_if_not_exists, str_to_list, get_cmd_args, load_json, benchmark
from model.common import DATASET_DIR, default_rayconfig_name, default_model, default_dataset, default_ray_trial_resources
from model.model_util import BINARY_CLF_MAP
from model.data_util import datagen, prepare_transpose_data, prepare_label_data
from recon.dataset_util import prep_dataset
from recon.split_util import pd_binary_clip


def exp(argv):
	cmd_arg_list = ['model=', 'dataset=', 'assets=']
	cmd_input = get_cmd_args(argv, cmd_arg_list, script_name=basename(__file__))
	mod_code = cmd_input['model='] if (cmd_input['model='] is not None) else default_model
	dataset_fname = cmd_input['dataset='] if (cmd_input['dataset='] is not None) else default_dataset
	assets = str_to_list(cmd_input['assets=']) if (cmd_input['assets='] is not None) else None

	mod = BINARY_CLF_MAP[mod_code]()
	mod_name = get_class_name(mod)
	dataset_dict = load_json(dataset_fname, dir_path=DATASET_DIR)
	dataset = prep_dataset(dataset_dict, assets=assets, filters_map=None)
	dataset_name = dataset_fname[:-JSON_SFX_LEN]
	exp_group_name = '{model},{dataset}'.format(model=mod_name, dataset=dataset_name)

	rayconfig = load_json(default_rayconfig_name, dir_path=MODEL_DIR)
	ray.init(**rayconfig['init'])

	logging.info('model: {}'.format(mod_name))
	logging.info('dataset: {} {} df(s)'.format(len(dataset['features']), dataset_name))
	logging.info('assets: {}'.format(str('all' if (assets==None) else ', '.join(assets))))
	logging.info('constructing {}...'.format(exp_group_name))
	exp_group = []

	for fpath, lpath, frec, lrec, fcol, lcol, feature, label in datagen(dataset, feat_prep_fn=prepare_transpose_data, label_prep_fn=prepare_label_data, how='ser_to_ser'):
		asset = fpath[0]
		assert(asset==lpath[0])
		exp_name = '{fdf}[{fcol}],{ldf}[{lcol}]'.format(fdf=frec.desc, fcol=fcol, ldf=lrec.desc, lcol=lcol)
		logging.info('experiment {exp}: {asset}'.format(exp=exp_name, asset=asset))
		exp_dir = REPORT_DIR +sep.join([asset, exp_group_name])
		makedir_if_not_exists(exp_dir)
		pos_label, neg_label = pd_binary_clip(label)

		pos_exp = Experiment('{},pos'.format(exp_name),
							run=mod.make_ray_objective(mod.make_const_data_objective(feature, pos_label)),
							stop=None,
							config=None,	# All config happens within model classes
							trial_resources=default_ray_trial_resources,
							num_samples=1,
							local_dir=exp_dir)
		neg_exp = Experiment('{},neg'.format(exp_name),
							run=mod.make_ray_objective(mod.make_const_data_objective(feature, neg_label)),
							stop=None,
							config=None,	# All config happens within model classes
							trial_resources=default_ray_trial_resources,
							num_samples=1,
							local_dir=exp_dir)
		exp_group.extend([pos_exp, neg_exp])

	logging.info('running {}...'.format(exp_group_name))
	algo = HyperOptSearch(mod.get_space(), max_concurrent=4, reward_attr='loss')
	trials = run_experiments(exp_group, search_alg=algo, verbose=True)

if __name__ == '__main__':
	with benchmark('time to finish') as b:
		exp(sys.argv[1:])
