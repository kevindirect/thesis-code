"""
Kevin Patel
"""
import sys
import os
import subprocess
import socket
import time
import logging

import pymongo

from common_util import makedir_if_not_exists, get_free_port, benchmark
from report.common import MONGOD_BIN, MONGOD_PORT, MONGOD_ARGS, MONGOD_CHECK_ATTEMPTS, DB_DIR


class MongoServer:
	"""
	ContextManager based wrapper on a local MongoDB server instance.
	Adapted from the MongoBox project: https://github.com/theorm/mongobox
	"""
	def __init__(self, mongod_bin=MONGOD_BIN, port=MONGOD_PORT, db_path=DB_DIR, directory_per_db=True, auth=False, scripting=False):
		self.mongod_bin = mongod_bin
		self.port = port or get_free_port()
		self.db_path = db_path
		self.directory_per_db = directory_per_db
		self.auth = auth
		self.scripting = scripting
		self.process = None

	def start(self):
		"""
		Start MongoDB.
		"""
		makedir_if_not_exists(self.db_path)

		args = [self.mongod_bin] + list(MONGOD_ARGS)
		args.extend(['--dbpath', self.db_path])
		args.extend(['--port', str(self.port)])

		self.log_path = os.path.join(self.db_path, 'log.log')
		args.extend(['--logpath', self.log_path])

		if (self.directory_per_db):
			args.append("--directoryperdb")

		if (self.auth):
			args.append("--auth")

		if (not self.scripting):
			args.append("--noscripting")

		self.process_args = args
		self.fnull = open(os.devnull, 'w')
		self.process = subprocess.Popen(args, stdout=self.fnull, stderr=subprocess.STDOUT, shell=False)
		self._wait_till_started()

	def _wait_till_started(self):
		attempts = 0
		while True:
			if (self.process.poll() is not None):  # The process has terminated
				with open(self.log_path) as log_file:
					raise SystemExit('MondgoDB failed to start:\n{}\n{}'.format(' '.join(self.process_args), log_file.read()))
			attempts += 1
			if (attempts > MONGOD_CHECK_ATTEMPTS):
				break
			s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			try:
				try:
					s.connect(('localhost', int(self.port)))
					return
				except (IOError, socket.error):
					time.sleep(0.25)
			finally:
				s.close()

		# MongoDB still does not accept connections. Killing it.
		self.stop()

	def stop(self):
		"""
		End MongoDB.
		"""
		if (self.process is None or self.process.poll() is not None):	# Process does not exist anymore
			return
		os.kill(self.process.pid, 9)
		self.process.wait()
		self.process = None
		self.fnull.close()
		self.fnull = None

	def running(self):
		return self.process is not None

	def get_client(self):
		"""
		Return client connection to this MongoDB server.
		"""
		try:
			return pymongo.MongoClient(host='localhost', port=self.port)  # version >=2.4
		except AttributeError:
			return pymongo.Connection(host='localhost', port=self.port)

	def __enter__(self):
		self.start()
		logging.info('Started MongoDB instance...')
		return self

	def __exit__(self, *args, **kwargs):
		self.stop()
		logging.info('Stopped MongoDB instance...')
