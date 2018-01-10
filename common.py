# Kevin Patel

from os import path, makedirs

makedir_if_not_exists = lambda dir_path: makedirs(dir_path) if not path.exists(dir_path) else None

def clean_cols(frame, clean_instr):
	if ("drop" in clean_instr):
		frame = frame.drop(clean_instr["drop"], axis=1, errors='ignore')
	if ("rename" in clean_instr):
		frame = frame.rename(columns=clean_instr["rename"])
	if ("col_prefix" in clean_instr):
		frame.columns = frame.columns.map(lambda s: str(clean_instr["col_prefix"] +s))
	return frame

month_num = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
			'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}
