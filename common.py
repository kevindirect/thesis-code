# Kevin Patel

from os import path, makedirs

makedir_if_not_exists = lambda dir_path: makedirs(dir_path) if not path.exists(dir_path) else None

month_num = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
			'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}
