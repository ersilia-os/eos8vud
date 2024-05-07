# the files moses train and moses val come from the data

import os
import sys
import csv
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm

from rdkit import RDLogger
import warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=Warning) 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

root = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(root, "..", "..", "checkpoints")
SQUID_DIR = os.path.join(root, "..", "code", "SQUID")
sys.path.append(SQUID_DIR)

from utils.general_utils import get_starting_seeds

AtomFragment_database = pd.read_pickle(os.path.join(CHECKPOINTS_DIR, 'data/MOSES2/MOSES2_training_val_AtomFragment_database.pkl'))
AtomFragment_database = AtomFragment_database.iloc[1:].reset_index(drop = True) # removing stop token from AtomFragment_database
fragment_library_atom_features = np.concatenate(AtomFragment_database['atom_features'], axis = 0).reshape((len(AtomFragment_database), -1))

bond_lookup = pd.read_pickle(os.path.join(CHECKPOINTS_DIR, 'data/MOSES2/MOSES2_training_val_bond_lookup.pkl'))
unique_atoms = np.load(os.path.join(CHECKPOINTS_DIR, 'data/MOSES2/MOSES2_training_val_unique_atoms.npy'))

smiles_list = []
with open(os.path.join(CHECKPOINTS_DIR, "data", "MOSES2", "MOSES2_train_smiles_split.csv"), "r") as f:
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        smiles_list += [r[1]]

with open(os.path.join(CHECKPOINTS_DIR, "data", "MOSES2", "MOSES2_val_smiles_split.csv"), "r") as f:
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        smiles_list += [r[1]]


mols = [Chem.MolFromSmiles(smi) for smi in tqdm(smiles_list)]

# keep only molecules with valid seed

print("Keeping only molecules that render a seed")
kept_smiles = []
for mol in tqdm(mols):
    select_seeds = get_starting_seeds(mol, AtomFragment_database, fragment_library_atom_features, unique_atoms, bond_lookup)
    if len(select_seeds) == 0:
        continue
    kept_smiles += [Chem.MolToSmiles(mol)]

print(len(kept_smiles))

with open(os.path.join(CHECKPOINTS_DIR, "reference_smiles.txt"), "w") as f:
    writer = csv.writer(f)
    for smiles in kept_smiles:
        writer.writerow([smiles])