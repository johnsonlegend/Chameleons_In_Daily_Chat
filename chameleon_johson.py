"""
Final Project for EECS 595
Implement word embedding method, part-of-speech pattern method on calculating sentence similarity.

@ chameleon_johson.py
@ johson
@ Nov 29, 2018
"""

import sys
import numpy as np
import pickle
from glob import glob
import gensim.downloader as api
from gensim.test.utils import common_texts, get_tmpfile


DEBUG = 1
TESTFILE = 3


def read_clean_data(fname):
	"Read in clean data."
	with open(fname, 'rb') as f:
		obj = pickle.load(f)

	return obj


def main():
	"Main function."
	clean_data_files = glob('TRN_output/*.pkl')
	if DEBUG:
		clean_data_files = clean_data_files[:TESTFILE]

	# Read
	for fname in clean_data_files:
		obj = read_clean_data(fname)
		print(len(obj))


if __name__ == '__main__':
	main()

