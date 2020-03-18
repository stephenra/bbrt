import pandas as pd
import selfies
import numpy as np
from selfies import encoder, decoder
from IPython import embed
import sys
import mmpa
import properties
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import os
import seaborn as sns

xdir ='/tigress/fdamani/mol-edit-output/onmt-logp04/preds/recurse_limit/src_train_900maxdiv_seeds/softmax_randtop5/maxdeltasim'

def remove_spaces(x):
	return x.replace(" ", "")

inp = pd.read_csv(xdir+'/1.csv', header=None, skip_blank_lines=False)
out = pd.read_csv(xdir+'/cands/2.csv', header=None, skip_blank_lines=False)

for i in range(inp.shape[0]):
	x = decoder(remove_spaces(''.join(inp.iloc[i])))
	x_molwt = properties.molwt(x)
	x_logp = properties.penalized_logp(x)
	x_qed = properties.qed(x)
	x_num_rotatable = properties.num_rotatable_bonds(x)
	x_num_h_donors = properties.num_h_donors(x)
	x_num_h_acceptors = properties.num_h_acceptors(x)
	if x == 'c1ccc(OCCCSc2ccccc2F)cc1':
		recs = out.iloc[i*100:i*100+100]
		smiles =[]
		sims = []
		logp= []
		qed=[]
		molwt = []
		num_rotatable = []
		num_h_donors = []
		num_h_acceptors = []
		for j in range(recs.shape[0]):
			ox = decoder(remove_spaces(''.join(recs.iloc[j])))
			if ox in smiles:
				continue
			local_sim = mmpa.similarity(ox, x)
			if local_sim is None:
				continue
			sims.append(mmpa.similarity(ox, x))
			smiles.append(ox)
			logp.append(properties.penalized_logp(ox))
			qed.append(properties.qed(ox))
			molwt.append(properties.molwt(ox))
			num_rotatable.append(properties.num_rotatable_bonds(ox))
			num_h_donors.append(properties.num_h_donors(ox))
			num_h_acceptors.append(properties.num_h_acceptors(ox))
		sims = np.array(sims)
		smiles = np.array(smiles)
		logp = np.array(logp)
		qed = np.array(qed)
		molwt = np.array(molwt)
		num_rotatable = np.array(num_rotatable)
		num_h_donors = np.array(num_h_donors)
		num_h_acceptors = np.array(num_h_acceptors)
		embed()