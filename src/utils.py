import argparse
import sys
import time
import numpy as np
import pandas as pd
from itertools import combinations
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
import re

import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt



def load_iter_data(start_iter, end_iter, data_dir):
    """
    Loads compound data from BBRT 
    
    input
    ----
    start_iter: index of starting iteration
    end_iter: index of ending iteration
    data_dir: path to data directory
    
    output
    ----
    cmpds: Pandas DataFrame of SMILES for each iteratino
    iter_start: List of SMILES for starting iteration
    iter_end: List of SMILES for ending iteration
    
    """
    dfs = {i: pd.read_csv(data_dir + '/scored_preds_{}.csv'.format(i),  header=None) for i in [start_iter, end_iter]}
    cmpds = pd.concat([dfs[start_iter], dfs[end_iter]], axis=1)
    for name in cmpds:
        cmpds.columns = ['Iteration {} SMILES'.format(name) for name in [start_iter, end_iter]]
    iter_start = cmpds.iloc[:, 0].tolist()
    iter_end = cmpds.iloc[:, 1].tolist()
    return cmpds, iter_start, iter_end


def smiles_to_fp(cmpds_list):
    """
    Converts SMILES to Morgan fingerprints
    
    input
    ----
    cmpds_list: list of compounds-as-SMILES
    
    output
    ----
    fp_list: list of RDKit bitvects
    """
    mol_list = []
    fp_list = []
    for smiles in cmpds_list:
        try:
            molecule = Chem.MolFromSmiles(smiles)
            mol_list.append(molecule)
            mol_list = [x for x in mol_list if x is not None]  # filter for un-decodable molecules
        except:
            print('ಠ_ಠ...Invalid SMILES : ', smiles)
    for mol in mol_list:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_list.append(fingerprint)
    return fp_list


def get_preds(start_smiles, end_smiles, model_path, cmpds):
    """
    Generates predicted potency values using fitted predictor
    
    input
    ----
    start_smiles: List of SMILES generated for starting iteration
    end_smiles: List of SMILES generated  dfor ending iteration
    cmpds: Pandas DataFrame of generated compound data
    
    output
    ----
    cmpds: Pandas DataFrame of predicted potencies for each iteration
    """
    fp_start = smiles_to_fp(start_smiles)
    fp_end = smiles_to_fp(end_smiles)
    clf = joblib.load(model_path + '/potency_predictor.pkl')
    start_time = time.time()
    pred_start = clf.predict(fp_start)
    pred_end = clf.predict(fp_end)
    print("Prediction time: ", round(time.time()-start_time, 3), "s")
    cmpds["Iteration 1 Predicted Potency"] = pd.Series(pred_start)
    cmpds["Iteration 20 Predicted Potency"] = pd.Series(pred_end)
    return cmpds


def plot_potency(iter_start, iter_end, model_path, data_dir, cmpds):
    """
    Generates predicted potency values using fitted predictor
    
    input
    ----
    iter_start: List of SMILES generated for starting iteration
    iter_end: List of SMILES generated for ending iteration
    model_path: path to potency predictor
    data_dir: path to data directory
    cmpds: Pandas DataFrame of generated compound data
    
    output
    ----
    potency_hist: Altair chart object; histogram of potency values by iteration
    """
    cmpds = get_preds(iter_start, iter_end, model_path, cmpds)
    potency_hist = alt.Chart(cmpds).transform_fold(
        ['Iteration 1 Predicted Potency', 'Iteration 20 Predicted Potency'],
        as_=['Iteration', 'Predicted Potency']
        ).mark_area(
            opacity=0.2,
            interpolate='step'
        ).encode(
            alt.X('Predicted Potency:Q', bin=True),
            alt.Y('count()', stack=None),
            alt.Color('Iteration:N')
        )
    potency_hist.display()
    potency_hist.save(data_dir + '/hist_potency_by_iteration.png')


def main():
    parser = argparse.ArgumentParser(description='Generate and visualize potency predictions')
    parser.add_argument('--data_dir', default='/data/bbrt/bpgm/output', type=str,
                      help='Path to data dir of BBRT output (default: /data/dir/bpgm/output)')
    parser.add_argument('--start_iter', default=0, type=int,
                      help='Index of starting iteration (default: 0)')
    parser.add_argument('--end_iter', default=29, type=int,
                      help='Index of ending iteration (default: 29)')
    parser.add_argument('--model_path', default='/data/bbrt/bpgm', type=str,
                     help='path to predictor (default: /data/bbrt/bpgm)')

    args = parser.parse_args()

    data_dir = args.data_dir
    start_iter = args.start_iter
    end_iter = args.end_iter
    model_path = args.model_path

    cmpds, iter_start, iter_end = load_iter_data(start_iter, end_iter, data_dir)
    plot_potency(iter_start, iter_end, cmpds)
    

if __name__ == "__main__":
    main()