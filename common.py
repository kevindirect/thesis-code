# Kevin Patel

from os import path, makedirs

makedir_if_not_exists = lambda dir_path: makedirs(dir_path) if not path.exists(dir_path) else None

month_num = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
			'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}


# Selects a subset of item_list as dictated by qualifier_dict
# returns the subset that satisfies: (QUALIFIER_1 OR QUALIFIER_2 OR ... OR QUALIFIER_N-1) AND NOT IN QUALIFIER_EXCLUDE
def get_subset(item_list, qualifier_dict):
	selected = []

	selected.extend(qualifier_dict['exact'])
	selected.extend([col for col in item_list if col.startswith(tuple(qualifier_dict['startswith']))])
	selected.extend([col for col in item_list if col.endswith(tuple(qualifier_dict['endswith']))])
	selected.extend([col for col in item_list if any(re.match(rgx, col) for rgx in qualifier_dict['regex'])])
	selected = filter(lambda col: col not in qualifier_dict['exclude'], selected)
	selected = list(dict.fromkeys(selected)) # Make a set (casting to dict keys retains order in Python 3.6+) and cast to list

	return selected
