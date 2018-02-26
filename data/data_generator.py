# Kevin Patel

import sys
from os import sep
from functools import reduce
from common import load_json, load_csv, makedir_if_not_exists, inner_join, get_subset
from common import DATA_DIR, default_joinfile, default_splitfile, default_accessfile

class DataGenerator:
	def __init__(self, joinfile=default_joinfile, splitfile=default_splitfile, accessfile=default_accessfile):
		# joinfile enumerates the available assets to run on
		self.assets = list(load_json(joinfile, dir_path=DATA_DIR).keys())

		# splitfile enumerates the available split groups and splits within them
		self.splits = load_json(splitfile, dir_path=DATA_DIR)

		# accessfile defines the group membership of each label group and the columns they can access
		# It defines the default data that will be furnished by the generator
		self.access = load_json(accessfile)

	def get_generator(self, asset_list=None, access_dict=None):
		asset_list = self.assets if (asset_list is None) else asset_list
		access_dict = self.access if (access_dict is None) else access_dict

		assert(all((asset in self.assets) for asset in asset_list))
		assert(all((split_group in self.splits.keys()) for split_group in access_dict.keys()))

		for split_group_name, split_group in access_dict.items():
			split_group_dir = DATA_DIR +split_group_name +sep

			for asset in asset_list:
				asset_dir = split_group_dir +asset +sep
				label_access_levels = split_group['#ASSET']

				for label_name, access_levels in label_access_levels.items():
					all_splits = [al['data_access'] for al in access_levels.values()]
					split_set = list(dict.fromkeys(reduce(lambda a,b: a+b, all_splits)))
					assert(all((split in self.splits[split_group_name]['#ASSET']) for split in split_set))
					split_dfs = {split: load_csv(str(split +'.csv'), dir_path=asset_dir) for split in split_set}
					label_df = load_csv(str(label_name +'.csv'), dir_path=asset_dir)

					for access_level_name, access_level in access_levels.items():
						labgroup_cols = get_subset(label_df.columns, access_level['label_group'])
						access_df = reduce(inner_join, (split_dfs[split] for split in access_level['data_access']))

						yield (split_group_name, asset, access_level_name, access_df, label_df[labgroup_cols])


def main(argv):
	dg = DataGenerator()
	dg.access.pop('hourly_mdl', None)
	# dg.access['hourly_mocl']['#ASSET']['label'].pop('TO_CLOSE', None)

	for tup in dg.get_generator():
		print('split group', tup[0])
		print('asset', tup[1])
		print('access level', tup[2])
		print(tup[3].head())
		print(tup[4].head())

if __name__ == '__main__':
	main(sys.argv[1:])

