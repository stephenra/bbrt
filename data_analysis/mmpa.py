'''
Generate training pairs using Matched molecular pair analysis
Find all pairs of compounds with tanimoto similarity > x and large differences in potency values
'''
import torch
import rdkit
import rdkit.Chem.QED as QED
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython import embed
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

def similarity(a, b):
	'''https://github.com/wengong-jin/iclr19-graph2graph/blob/master/props/properties.py'''

	if a is None or b is None: 
		return None
	amol = Chem.MolFromSmiles(a)
	bmol = Chem.MolFromSmiles(b)
	if amol is None or bmol is None:
		return None

	fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
	fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
	return DataStructs.TanimotoSimilarity(fp1, fp2)

def mgn_fgpt(a):

	if a is None:
		return 0.0
	amol = Chem.MolFromSmiles(a)
	if amol is None:
		return 0.0

	fp = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
	return fp

def population_diversity(x):
	'''take a set of compounds compute pairwise diversity measures
	diversity = 1 - sim(a,b)
	'''
	diversity = []
	for i in range(len(x)):
		for j in range(len(x)):
			if j > i:
				if similarity(x[i], x[j]) is None:
					continue
				diversity.append(1.0 - similarity(x[i], x[j]))
	return diversity

if __name__ == "__main__":
	dat = pd.read_csv('/data/potency_dataset_with_props.csv')
	dat = dat.sample(frac=1).reset_index(drop=True)
	structure = dat['Structure']
	pot = -np.log10(dat['pot_uv'])
	similarity_scores = []

	# decide on transform
	# inactive structure has potency value of -3.
	inactive_structs = structure[pot == -3.].values
	# limit to 75 percentile potency threshold (conditioned on non-zero potency)
	pot_thresh = np.percentile(pot[pot > -3.], 50)
	active_structs = structure[pot > pot_thresh].values
	active_structs_pot = pot[pot > pot_thresh].values
	for idx, sx in enumerate(inactive_structs):
		for ind,ax in enumerate(active_structs):
			pot_val = active_structs_pot[ind]
			similarity_scores.append((sx, ax, similarity(sx, ax), pot_val))
		if idx % 5000 == 0:
			torch.save(pd.DataFrame(similarity_scores), '/data/pytorch_obj/mmpa_similarity_full.pth')
	torch.save(pd.DataFrame(similarity_scores), '/data/pytorch_obj/mmpa_similarity_full.pth')