# Kevin Patel

import numpy as np
import pandas as pd
import sys
import getopt
from os import getcwd, sep, path, makedirs, pardir
import json
from functools import reduce
sys.path.insert(0, path.abspath(pardir))
from common_util import makedir_if_not_exists, inner_join, load_csv, get_subset


class DataGenerator:
	pfx = getcwd() +sep

	def __init__(self, joinfile='join.json', splitfile='split.json', accessfile='access.json'):
		# joinfile enumerates the available equities to run on
		if (path.isfile(DataGenerator.pfx +joinfile)):
			with open(DataGenerator.pfx +joinfile) as json_data:
				self.equities = list(json.load(json_data).keys())
		else:
			print(joinfile, 'must be present in the following directory:', DataGenerator.pfx)
			sys.exit(2)

		# splitfile enumerates the available split groups and splits within them
		if (path.isfile(DataGenerator.pfx +splitfile)):
			with open(DataGenerator.pfx +splitfile) as json_data:
				self.splits = json.load(json_data)
		else:
			print(splitfile, 'must be present in the following directory:', DataGenerator.pfx)
			sys.exit(2)

		# accessfile defines the group membership of each label group and the columns they can access
		# It defines the default data that will be furnished by the generator
		if (path.isfile(DataGenerator.pfx +accessfile)):
			with open(DataGenerator.pfx +accessfile) as json_data:
				self.access = json.load(json_data)
		else:
			print(accessfile, 'must be present in the current directory')
			sys.exit(2)

	def get_generator(self, asset_list=None, access_dict=None):
		if (asset_list is None):
			asset_list = self.equities
		if (access_dict is None):
			access_dict = self.access
		assert(all((asset in self.equities) for asset in asset_list))
		assert(all((split_group in self.splits.keys()) for split_group in access_dict.keys()))

		for split_group_name, split_group in access_dict.items():
			split_group_dir = DataGenerator.pfx +split_group_name +sep

			for equity in asset_list:
				equity_dir = split_group_dir +equity +sep
				for label_name, access_levels in split_group['#ASSET'].items():
					label_df = load_csv(equity_dir +label_name +'.csv')
					all_labels_used = []
					catch_all_labgroup = None

					for level_name, access_level in access_levels.items():
						assert(all((colgroup in self.splits[split_group_name]['#ASSET']) for colgroup in access_level['column_access']))
						if (isinstance(access_level['label_group'], str)):
							# Special label groups
							if (access_level['label_group'] == 'catch_all'):
								catch_all_labgroup = level_name
							continue
						labgroup_qual = access_level['label_group']
						labgroup_cols = get_subset(label_df.columns, labgroup_qual)
						all_labels_used.extend(labgroup_cols)

						colgroups = access_level['column_access']
						colgroups_df = reduce(inner_join, map(lambda a: load_csv(equity_dir +a +'.csv'), colgroups))

						yield (split_group_name, equity, level_name, colgroups_df, label_df[labgroup_cols])

					# Special label groups are yielded at the end:
					if (isinstance(catch_all_labgroup, str)):
						level_name = catch_all_labgroup
						access_level = access_levels[level_name]
						assert(all((colgroup in self.splits[split_group_name]['#ASSET']) for colgroup in access_level['column_access']))

						labgroup_qual = {
							"exact": [],
							"startswith": [],
							"endswith": [],
							"regex": [],
							"exclude": {
								"exact": all_labels_used,
								"startswith": [],
								"endswith": [],
								"regex": [],
								"exclude": None
							}
						}
						labgroup_cols = get_subset(label_df.columns, labgroup_qual)

						colgroups = access_level['column_access']
						colgroups_df = reduce(inner_join, map(lambda a: load_csv(equity_dir +a +'.csv'), colgroups))

						yield (split_group_name, equity, level_name, colgroups_df, label_df[labgroup_cols])


def main(argv):
	dg = DataGenerator()
	dg.access.pop('hourly_mdl', None)

	for tup in dg.get_generator():
		print('split group', tup[0])
		print('equity', tup[1])
		print('access level', tup[2])

if __name__ == '__main__':
	main(sys.argv[1:])

