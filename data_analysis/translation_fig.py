'''read in translation_example file
first line src smiles string
rest of the lines: predictions


arg1: src_directory (where all translation_example.csv files are)
arg2: output_dir

example: python translation_fig.py \
	/tigress/fdamani/mol-edit-output/onmt-logp04/output/output-firstiter-baseline-20beam-model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_99000/example_outputs/data 
	/tigress/fdamani/mol-edit-output/onmt-logp04/output/output-firstiter-baseline-20beam-model-brnnenc-rnndec-2layer-600wordembed-600embed-shareembedding-mlpattention-adamoptim_step_99000/example_outputs
'''

import numpy as np
import rdkit
import pandas as pd
import properties
from IPython import embed
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing
import sys
import os
import shutil
src_dir = sys.argv[1]
output_dir = sys.argv[2]

if os.path.exists(output_dir+'/output_figs'):
	print("DIRECTORY EXISTS. Continue to delete.")
	shutil.rmtree(output_dir+'/output_figs')
os.mkdir(output_dir+'/output_figs')
for file in os.listdir(src_dir):
	if file.endswith(".csv"):
		name = file.split(".csv")[0]
		data = pd.read_csv(src_dir+"/"+file,header=None)
		src_mol = Chem.MolFromSmiles(data.iloc[0].values[0])
		tgt_mols = [Chem.MolFromSmiles(data.iloc[i].values[0]) for i in range(1, data.shape[0])]
		src_legend = [str('%.3f'%(properties.penalized_logp(data.iloc[0].values[0])))]
		tgt_legend = [str('%.3f'%(properties.penalized_logp(data.iloc[i].values[0]))) for i in range(1, data.shape[0])]
		# save src mol
		img = Draw.MolsToGridImage([src_mol], molsPerRow=1, subImgSize=(500,500), legends=src_legend)
		img.save(output_dir+"/output_figs/"+name+"_src.png")

		# save tgt mols
		img = Draw.MolsToGridImage(tgt_mols, molsPerRow=5, subImgSize=(500,500), legends=tgt_legend)
		img.save(output_dir+"/output_figs/"+name+"_tgt.png")
	print(file)