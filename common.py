# Kevin Patel

from os import path, makedirs

makedir_if_not_exists = lambda dir_path: makedirs(dir_path) if not path.exists(dir_path) else None





