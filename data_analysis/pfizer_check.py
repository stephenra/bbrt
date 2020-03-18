import pandas as pd
import rdkit
from rdkit import Chem
import properties
from IPython import embed

x = pd.read_csv("cmpds_pfizer_check_unique.csv", header=None)
count =0
for i in range(x.shape[0]):
	val = properties.penalized_logp(x.iloc[i].values[0])
	if val > 3.143 and val < 3.148:
		embed()
		count+=1
print(count)
embed()