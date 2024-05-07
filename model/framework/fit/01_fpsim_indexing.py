import os
import csv
from FPSim2.io import create_db_file

root = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.abspath(os.path.join(root, "..", "..", "checkpoints"))

print("Reading SMILES")
smiles_list = []
with open(os.path.join(CHECKPOINTS_DIR, "reference_smiles.txt"), "r") as f:
    reader = csv.reader(f)
    for i, r in enumerate(reader):
        smiles_list += [[r[0], i]]

# Atom-pair fingerprints capture 3D shape quite well (see: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00445-4)
print("Atom-pair fingerprints")
h5_file = os.path.join(CHECKPOINTS_DIR, "atompair_fps.h5")
if os.path.exists(h5_file):
    os.remove(h5_file)
create_db_file(smiles_list, h5_file, 'AtomPair', {'nBits': 2048})

# We also do Morgan fingerprints
print("Morgan fingerprints")
h5_file = os.path.join(CHECKPOINTS_DIR, "morgan_fps.h5")
if os.path.exists(h5_file):
    os.remove(h5_file)
create_db_file(smiles_list, h5_file, 'Morgan', {'radius': 2, 'nBits': 2048})

print("Done!")