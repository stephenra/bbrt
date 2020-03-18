'''compute histogram over tokens for training, held-out-test, and test from jin 2018'''
import torch
import numpy as np
import pandas as pd
import sys
import properties
import mmpa
import selfies
from selfies import encoder, decoder

from IPython import embed

def remove_spaces(x):
	return x.replace(" ", "")

training_data_src = pd.read_csv(sys.argv[1], header=None)
training_data_tgt = pd.read_csv(sys.argv[2], header=None)
held_out_test_src = pd.read_csv(sys.argv[3], header=None)
held_out_test_tgt = pd.read_csv(sys.argv[4], header=None)
#jin_test = pd.read_csv(sys.argv[3], header=None)

held_out_test_sim = []
matching_inds = []
for i in range(held_out_test_src.shape[0]):
	sx = decoder(remove_spaces(''.join(held_out_test_src.iloc[i])))
	sims = []

	# check if held out test sample is in training data
	dup = training_data_src[training_data_src==held_out_test_src.iloc[i]].dropna()
	if len(dup) > 0:
		matching_inds.append(i)


	print(i)
	# for j in range(training_data_src.shape[0]):
	# 	ax = decoder(remove_spaces(''.join(training_data_src.iloc[j])))
	# 	sim = mmpa.similarity(sx, ax)
	# 	sims.append(mmpa.similarity(sx, ax))
	# 	if sim == 1.0:
	# 		print('match found.')
	# 		print('printing corresponding targets...')
	# 		sx_tgt = decoder(remove_spaces(''.join(held_out_test_tgt.iloc[i])))
	# 		ax_tgt = decoder(remove_spaces(''.join(training_data_tgt.iloc[j]))) 
	# 		print(sx_tgt, ax_tgt)
	# 		break
	# sims = np.array(sims)
	# held_out_test_sim.append(np.max(sims))
	# print(i)
	# matching_inds.append(i)


embed()