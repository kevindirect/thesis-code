Takes in raw data, joins it, and splits it into split groups which are dumped to the filesystem.
Supplies a class that contains generator functions for data tuples.

produce_data.py: joins raw data and splits them into split groups
generate_data.py: supplies data in tuples of (access_level_df, label_group_df).
	The access_level_df contains the allowable data that can be used to predict any label in label_group_df.
	