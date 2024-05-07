# imports
import os
import csv
import sys

root = os.path.dirname(os.path.abspath(__file__))
squid_path = os.path.join(root, "SQUID")
sys.path.append(squid_path)
from do_one import generate_molecules

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

N_MOLECULES = 100

# current file directory
root = os.path.dirname(os.path.abspath(__file__))

# read SMILES from .csv file, assuming one column with header
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    smiles_list = [r[0] for r in reader]

header = ["smi-{0}".format(str(i).zfill(3)) for i in range(N_MOLECULES)]

# run model
outputs = []
for smiles in smiles_list:
    outputs += [generate_molecules(smiles)]

#check input and output have the same length
input_len = len(smiles_list)
output_len = len(outputs)
assert input_len == output_len

# write output in a .csv file
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(header)  # header
    for o in outputs:
        writer.writerow(o)
