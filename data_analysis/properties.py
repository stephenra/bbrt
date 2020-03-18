'''
property scoring
'''
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import rdkit.Chem.QED as QED
import sascorer
import networkx as nx

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
def morgan_fpt(a):
    if a is None:
        return 0.0
    amol = Chem.MolFromSmiles(a)
    if amol is None:
        return 0.0
    fp = AllChem.GetMorganFingerprint(amol, 2)

    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=512, useChirality=False)
    return fp, fp1

def num_rotatable_bonds(s):
	mol = Chem.MolFromSmiles(s)
	if mol is None:
		return None
	return rdkit.Chem.Lipinski.NumRotatableBonds(mol)

def num_h_donors(s):
	mol = Chem.MolFromSmiles(s)
	if mol is None:
		return None
	return rdkit.Chem.Lipinski.NumHDonors(mol)

def num_h_acceptors(s):
	mol = Chem.MolFromSmiles(s)
	if mol is None:
		return None
	return rdkit.Chem.Lipinski.NumHAcceptors(mol)

def molwt(s):
	mol = Chem.MolFromSmiles(s)
	if mol is None:
		return 0.0
	return rdkit.Chem.Descriptors.ExactMolWt(mol)


def qed(s):
	'''https://github.com/wengong-jin/iclr19-graph2graph/blob/master/props/properties.py'''
	if s is None:
		return 0.0
	mol = Chem.MolFromSmiles(s)
	if mol is None:
		return 0.0
	# try except to avoid Sanitization error 
	# ex: ValueError: Sanitization error: Explicit valence for atom # 19 O, 3, is greater than permitted
	try:
		qd = QED.qed(mol)
		return qd
	except:
		return 0.0

def sa(s):
	mol = Chem.MolFromSmiles(s)
	SA = sascorer.calculateScore(mol)
	return SA

def penalized_logp(s):
	'''https://github.com/wengong-jin/iclr19-graph2graph/blob/master/props/properties.py'''
	if s is None: return None #-100.0
	mol = Chem.MolFromSmiles(s)
	if mol is None: return None #-100.0

	logP_mean = 2.4570953396190123
	logP_std = 1.434324401111988
	SA_mean = -3.0525811293166134
	SA_std = 0.8335207024513095
	cycle_mean = -0.0485696876403053
	cycle_std = 0.2860212110245455

	log_p = Descriptors.MolLogP(mol)
	SA = -sascorer.calculateScore(mol)

	# cycle score
	cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
	if len(cycle_list) == 0:
		cycle_length = 0
	else:
		cycle_length = max([len(j) for j in cycle_list])
	if cycle_length <= 6:
		cycle_length = 0
	else:
		cycle_length = cycle_length - 6
	cycle_score = -cycle_length

	normalized_log_p = (log_p - logP_mean) / logP_std
	normalized_SA = (SA - SA_mean) / SA_std
	normalized_cycle = (cycle_score - cycle_mean) / cycle_std
	return normalized_log_p + normalized_SA + normalized_cycle

