"""Take as input train_pairs.txt file.
	
- separate src/target
- convert to selfies
- save src/target separately with space-separated tokens, one compound per line

Example command: python process_data.py ../data/logp04/train_pairs.txt ../data/logp04"""

import torch
import numpy as np
import pandas as pd
import sys
import selfies
import re

from selfies import encoder, decoder
from IPython import embed, display

def read_data(file, selfies=True):
	data = pd.read_csv(file, delimiter=' ', header=None)
	#data = data.head(1000)
	# return all rows in random order.
	data = data.sample(frac=1).reset_index(drop=True)
	src = smiles_to_selfies(data[0])
	tgt = smiles_to_selfies(data[1])
	pairs = pd.DataFrame([src, tgt]).T
	pairs = pairs.dropna()
	src, tgt = pairs[0], pairs[1]
	return src, tgt

def smiles_to_selfies(x, token_sep=True):
	"""smiles to selfies
	if token_sep=True -> return spaces between each token"""
	output = []
	for i in range(x.shape[0]):
		ax = encoder(x[i])
		if ax != -1:
			if token_sep:
				sx = re.findall(r"\[[^\]]*\]", ax)
				ax = ' '.join(sx)
			output.append(ax)
		else:
			output.append('NaN')
	return output

def train_test_split(X, Y, percent_train=.7, test=False):
	'''train test split: assume data has already been permuted'''
	if test:
		n_samples = len(X)
		train_ind = int(n_samples * percent_train)
		valid_ind = int(train_ind + ((n_samples-train_ind)/2.0))
		
		X_train, Y_train = X[0:train_ind], Y[0:train_ind]
		X_valid, Y_valid = X[train_ind:valid_ind], Y[train_ind:valid_ind]
		X_test, Y_test = X[valid_ind:], Y[valid_ind:]
		embed()
		return X_train, Y_train, X_valid, Y_valid, X_test, Y_test
	else:
		n_samples = len(X)
		train_ind = int(n_samples * percent_train)
		valid_ind = int(train_ind + ((n_samples-train_ind)/2.0))
		
		X_train, Y_train = X[0:train_ind], Y[0:train_ind]
		X_valid, Y_valid = X[train_ind:], Y[train_ind:]

		return X_train, Y_train, X_valid, Y_valid

if __name__ == '__main__':
	file = sys.argv[1]
	output_dir = sys.argv[2]
	src, tgt = read_data(file)

	test = False
	# when using ONMT we want our validation to have 5k
	#num_test = 360.0
	num_test = 5000.0
	#num_test = 50.0
	#percent_test = float(num_test / src.shape[0])
	percent_test = .1
	if test:
		percent_train = 1 - 2 * percent_test
	else:
		percent_train = 1 - percent_test
	print('percent train: ', percent_train, ' percent_test: ', percent_test)
	if test:
		src_train, tgt_train, src_valid, tgt_valid, src_test, tgt_test = train_test_split(src, tgt, percent_train, test)
	else:
		src_train, tgt_train, src_valid, tgt_valid = train_test_split(src, tgt, percent_train, test)

	# save to dir
	src_train.to_csv(output_dir+'/src_train.csv',index=None, header=None)
	tgt_train.to_csv(output_dir+'/tgt_train.csv',index=None, header=None)
	src_valid.to_csv(output_dir+'/src_valid.csv',index=None, header=None)
	tgt_valid.to_csv(output_dir+'/tgt_valid.csv',index=None, header=None)
	if test:
		src_test.to_csv(output_dir+'/src_test.csv',index=None, header=None)
		tgt_test.to_csv(output_dir+'/tgt_test.csv',index=None, header=None)