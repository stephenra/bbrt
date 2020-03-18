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
import mmpa

x = pd.read_csv(sys.argv[1], header=None)
seed = x.iloc[0].values[0]
for i in range(1, x.shape[0]):
	print(mmpa.similarity(seed, x.iloc[i].values[0]))