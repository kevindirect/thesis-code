# Kevin Patel

from os import path, makedirs

makedir_if_not_exists = lambda dir_path: makedirs(dir_path) if not path.exists(dir_path) else None

month_num = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
			'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}

# Selects a subset of item_list as dictated by qualifier_dict
# returns the subset that satisfies:
# IN(QUALIFIER_1 OR QUALIFIER_2 OR ... OR QUALIFIER_N-1) AND NOT IN(EXCLUDE_1 OR EXCLUDE_2 OR ... OR EXCLUDE_N-1)
def get_subset(item_list, qualifier_dict):
	selected = []

	selected.extend(qualifier_dict['exact'])
	selected.extend([col for col in item_list if col.startswith(tuple(qualifier_dict['startswith']))])
	selected.extend([col for col in item_list if col.endswith(tuple(qualifier_dict['endswith']))])
	selected.extend([col for col in item_list if any(re.match(rgx, col) for rgx in qualifier_dict['regex'])])
	if (qualifier_dict['exclude'] is not None):
		exclude_fn = lambda col: col not in get_subset(item_list, qualifier_dict['exclude'])
		selected = filter(exclude_fn, selected)

	return list(dict.fromkeys(selected)) # Remove dups (casting to dict keys retains order in Python 3.6+) and cast to list
