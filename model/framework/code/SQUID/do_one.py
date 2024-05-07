import torch
import torch_geometric
import torch_scatter
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.rdMolTransforms
import networkx as nx
import random
from tqdm import tqdm
from rdkit.Chem import rdMolTransforms
import itertools
import os
import pickle

import torch.nn as nn
import torch.nn.functional as F
from models.vnn.models.vn_layers import *
from models.vnn.models.utils.vn_dgcnn_util import get_graph_feature

from utils.general_utils import *

from models.EGNN import *
from models.models import *

from FPSim2 import FPSim2Engine
import json

root = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(root, "..", "..", "..", "checkpoints")


interpolate_to_GNN_prior = 1.0 # 'prior'
stop_threshold = 0.01
variational_GNN_factor = 1.0
ablateEqui = False

repetitions = 20
total_evaluations = 10 


use_artificial_mols = True

AtomFragment_database = pd.read_pickle(os.path.join(CHECKPOINTS_DIR, 'data/MOSES2/MOSES2_training_val_AtomFragment_database.pkl'))
AtomFragment_database = AtomFragment_database.iloc[1:].reset_index(drop = True) # removing stop token from AtomFragment_database
fragment_library_atom_features = np.concatenate(AtomFragment_database['atom_features'], axis = 0).reshape((len(AtomFragment_database), -1))

bond_lookup = pd.read_pickle(os.path.join(CHECKPOINTS_DIR, 'data/MOSES2/MOSES2_training_val_bond_lookup.pkl'))
unique_atoms = np.load(os.path.join(CHECKPOINTS_DIR, 'data/MOSES2/MOSES2_training_val_unique_atoms.npy'))


print(bond_lookup)
print(unique_atoms)

if not ablateEqui:
    model_3D_PATH = os.path.join(root, 'trained_models/graph_generator.pt')
    rocs_model_3D_PATH = os.path.join(root, 'trained_models/scorer.pt')
else:
    model_3D_PATH = os.path.join(root, 'trained_models/graph_generator_ablateEqui.pt')
    rocs_model_3D_PATH = os.path.join(root, 'trained_models/scorer_ablateEqui.pt')


# FPSim2 similarity searches
fp0_filename = os.path.join(CHECKPOINTS_DIR, "atompair_fps.h5")
fp0 = FPSim2Engine(fp0_filename)
fp1_filename = os.path.join(CHECKPOINTS_DIR, "morgan_fps.h5")
fp1 = FPSim2Engine(fp1_filename)

with open(os.path.join(CHECKPOINTS_DIR, "reasonable_fp_cutoffs.json"), "r") as f:
    search_cutoffs = json.load(f)

# HYPERPARAMETERS for 3D graph generator
pointCloudVar = 1. / (12. * 1.7) 

model_3D = Model_Point_Cloud_Switched(
    input_nf = 45, 
    edges_in_d = 5, 
    n_knn = 5, 
    conv_dims = [32, 32, 64, 128], 
    num_components = 64, 
    fragment_library_dim = 64, 
    N_fragment_layers = 3, 
    append_noise = False, 
    N_members = 125 - 1, 
    EGNN_layer_dim = 64, 
    N_EGNN_layers = 3, 
    output_MLP_hidden_dim = 64, 
    pooling_MLP = False, 
    shared_encoders = False, 
    subtract_latent_space = True,
    variational = False,
    variational_mode = 'inv', # not used
    variational_GNN = True,
    
    mix_node_inv_to_equi = True,
    mix_shape_to_nodes = True,
    ablate_HvarCat = False,
    
    predict_pairwise_properties = False,
    predict_mol_property = False,
    
    ablateEqui = ablateEqui,
    
    old_EGNN = False,
    
).float()

model_3D.load_state_dict(torch.load(model_3D_PATH, map_location=next(model_3D.parameters()).device), strict = True)
model_3D.eval()


# HYPERPARAMETERS for ROCS scorer
rocs_pointCloudVar = 1. / (12. * 1.7) 

rocs_model_3D = ROCS_Model_Point_Cloud(
    input_nf = 45, 
    edges_in_d = 5, 
    n_knn = 10, 
    conv_dims = [32, 32, 64, 128], 
    num_components = 64, 
    fragment_library_dim = 64,
    N_fragment_layers = 3, 
    append_noise = False, 
    N_members = 125 - 1, 
    EGNN_layer_dim = 64, 
    N_EGNN_layers = 3, 
    output_MLP_hidden_dim = 64, 
    pooling_MLP = False, 
    shared_encoders = False, 
    subtract_latent_space = True,
    variational = False,
    variational_mode = 'inv', # not used
    variational_GNN = False,
    
    mix_node_inv_to_equi = True,
    mix_shape_to_nodes = True,
    ablate_HvarCat = False,
    
    ablateEqui = ablateEqui,
    
    old_EGNN = False,
    
).float()

rocs_model_3D.load_state_dict(torch.load(rocs_model_3D_PATH, map_location=next(rocs_model_3D.parameters()).device), strict = True)
rocs_model_3D.eval()


# muting noisy warnings
from rdkit import RDLogger
import warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=Warning) 
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

import csv
import subprocess
import tempfile
from rdkit import Chem

tmp_folder = tempfile.mkdtemp(prefix="ersilia-")
python_path = sys.executable

N_CONFORMERS = 5

def prepare_molecule(smiles, k):
    smi_file = os.path.join(tmp_folder, "molecule.csv")
    with open(smi_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])
        writer.writerow([smiles])
    sdf3d_file = os.path.join(tmp_folder, "molecules3d.sdf")
    cmd = "{0} {1}/../smi3d/smi3d.py -i {2} -o {3} -n {4}".format(python_path, root, smi_file, sdf3d_file, k)
    print(cmd)
    subprocess.Popen(cmd, shell=True).wait()
    supplier = Chem.SDMolSupplier(sdf3d_file)
    mols = []
    for mol in supplier:
        mols += [mol]
    return mols


def is_seedable(smiles):
    mol = Chem.MolFromSmiles(smiles)
    select_seeds = get_starting_seeds(mol, AtomFragment_database, fragment_library_atom_features, unique_atoms, bond_lookup)
    if len(select_seeds) == 0:
        return False
    else:
        return True


def get_seedable_smiles(smiles):
    if is_seedable(smiles):
        return smiles
    print("molecule is not seedable")
    query = smiles
    results = fp0.similarity(query, search_cutoffs["atompair"], n_workers=1)
    print(results)


def generate_molecules(smiles):

    smiles = get_seedable_smiles(smiles)

    file_idx = 0
 
    reference_mols_list = []
    unaligned_mols_list = []
    
    mols = prepare_molecule(smiles, N_CONFORMERS)

    for m_idx, m_ in enumerate(mols):

        print(m_)
        
        seed = 0
        random.seed(seed)
        np.random.seed(seed = seed)
        torch.manual_seed(seed)
        
        m = deepcopy(m_)
        
        mol = deepcopy(m)
        xyz = np.array(mol.GetConformer().GetPositions())
        center_of_mass = np.sum(xyz, axis = 0) / xyz.shape[0]
        xyz_centered = xyz - center_of_mass
        for i in range(0, mol.GetNumAtoms()):
            x,y,z = xyz_centered[i]
            mol.GetConformer().SetAtomPosition(i, Point3D(x,y,z))
        
        m = deepcopy(mol)
        
        mol_target = deepcopy(m)
        ring_fragments = get_ring_fragments(mol_target)
        all_possible_seeds = get_all_possible_seeds(mol_target, ring_fragments)
        terminal_seeds = filter_terminal_seeds(all_possible_seeds, mol_target)
        
        select_seeds = get_starting_seeds(mol_target, AtomFragment_database, fragment_library_atom_features, unique_atoms, bond_lookup)
        print(select_seeds)
        
        if len(select_seeds) == 0:
            continue
        
        random_seed_selection = random.randint(0, len(select_seeds) - 1)
        select_seeds = [select_seeds[random_seed_selection]] * repetitions
        
        repeated_rocs = []
        repeated_tanimoto = []

        reference_mols = []
        unaligned_mols = []
        
        
        for seed in select_seeds:

            mol = deepcopy(m)
            
            # extracting starting seed and preparing to generate
            
            frame_generation, frame_rocs = get_frame_terminalSeeds(mol, seed, AtomFragment_database, include_rocs = True)
            positions = list(frame_rocs.iloc[0].positions_before)
            start = 0
            for i in range(len(frame_generation)):
                if (set(frame_generation.iloc[i].partial_graph_indices) == set(positions)) & (frame_generation.iloc[i].next_atom_index == -1):
                    start = i + 1
                    break
            
            if len(frame_generation.iloc[0].partial_graph_indices) == 1: # seed is a terminal ATOM
                terminalSeed_frame = frame_generation.iloc[0:start].reset_index(drop = True)
                        
                sequence = get_ground_truth_generation_sequence(terminalSeed_frame, AtomFragment_database, fragment_library_atom_features)
                    
                mol = deepcopy(terminalSeed_frame.iloc[0].rdkit_mol_cistrans_stereo)
                partial_indices = deepcopy(terminalSeed_frame.iloc[0].partial_graph_indices_sorted)
                
                final_partial_indices = deepcopy(terminalSeed_frame.iloc[-1].partial_graph_indices_sorted)
                ring_fragments = get_ring_fragments(mol)
                add_to_partial = [list(f) for p in final_partial_indices for f in ring_fragments if p in f]
                add_to_partial = [item for sublist in add_to_partial for item in sublist]
                final_partial_indices = list(set(final_partial_indices).union(add_to_partial))
                    
                queue_indices = deepcopy(terminalSeed_frame.iloc[0].focal_indices_sorted)
                
                _, seed_mol, queue, positioned_atoms_indices, atom_to_library_ID_map, _, _, _ = generate_seed_from_sequence(sequence, mol, partial_indices, queue_indices, AtomFragment_database, unique_atoms, bond_lookup, stop_after_sequence = True)
        
                seed_node_features = getNodeFeatures(seed_mol.GetAtoms())
                
                for k in atom_to_library_ID_map:
                    seed_node_features[k] = AtomFragment_database.iloc[atom_to_library_ID_map[k]].atom_features
                    
                G = get_substructure_graph(mol, final_partial_indices)
                G_seed = get_substructure_graph(seed_mol, list(range(0, seed_mol.GetNumAtoms())), node_features = seed_node_features)
                nm = nx.algorithms.isomorphism.generic_node_match(['atom_features'], [None], [np.allclose])
                em = nx.algorithms.isomorphism.numerical_edge_match("bond_type", 1.0)
                GM = nx.algorithms.isomorphism.GraphMatcher(G, G_seed, node_match = nm, edge_match = em)
                assert GM.is_isomorphic()
                idx_map = GM.mapping
                    
            else: # seed is a terminal FRAGMENT
                partial_indices = deepcopy(frame_generation.iloc[0].partial_graph_indices_sorted)
                final_partial_indices = partial_indices
                seed_mol = generate_conformer(get_fragment_smiles(mol, partial_indices))
                idx_map = get_reindexing_map(mol, partial_indices, seed_mol)
                positioned_atoms_indices = sorted([idx_map[f] for f in final_partial_indices])
                
                atom_to_library_ID_map = {} # no individual atoms yet generated
                queue = [0] # 0 can be considered the focal root node
        
            for i in final_partial_indices:
                x,y,z = mol.GetConformer().GetPositions()[i]
                seed_mol.GetConformer().SetAtomPosition(idx_map[i], Point3D(x,y,z)) 
            
            
            
            starting_queue = deepcopy(queue)
            try:
                _, updated_mol, _, _, _, N_rocs_decisions, _, _, _, chirality_scored = generate_3D_mol_from_sequence(
                    sequence = [], 
                    partial_mol = deepcopy(seed_mol), 
                    mol = deepcopy(mol_target), 
                    positioned_atoms_indices = deepcopy(positioned_atoms_indices), 
                    queue = starting_queue, 
                    atom_to_library_ID_map = deepcopy(atom_to_library_ID_map), 
                    model = model_3D, 
                    rocs_model = rocs_model_3D,
                    AtomFragment_database = AtomFragment_database,
                    unique_atoms = unique_atoms, 
                    bond_lookup = bond_lookup,
                    N_points = 5, 
                    N_points_rocs = 5,
                    stop_after_sequence = False,
                    mask_first_stop = False,
                    stochastic = False, 
                    chirality_scoring = True,
                    stop_threshold = stop_threshold,
                    steric_mask = True,
                    
                    variational_factor_equi = 0.0,
                    variational_factor_inv = 0.0, 
                    interpolate_to_prior_equi = 0.0,
                    interpolate_to_prior_inv = 0.0, 
                    
                    use_variational_GNN = True, 
                    variational_GNN_factor = variational_GNN_factor, 
                    interpolate_to_GNN_prior = interpolate_to_GNN_prior, 
                    
                    rocs_use_variational_GNN = False, 
                    rocs_variational_GNN_factor = 0.0, 
                    rocs_interpolate_to_GNN_prior = 0.0,
                    
                    pointCloudVar = pointCloudVar, 
                    rocs_pointCloudVar = rocs_pointCloudVar,
                )
                
                pred_rocs = get_ROCS(torch.tensor(np.array(updated_mol.GetConformer().GetPositions())), torch.tensor(np.array(mol.GetConformer().GetPositions())))
                
                tanimoto = rdkit.DataStructs.FingerprintSimilarity(*[rdkit.Chem.RDKFingerprint(x) for x in [mol, updated_mol]])
                    
                reference_mols.append(mol)
                unaligned_mols.append(updated_mol)
                
                repeated_rocs.append(pred_rocs.item())
                repeated_tanimoto.append(tanimoto)
                
            except Exception as e:
                print(f'error during 3D generation -- {e}')
                continue
        
        print(f'Encoded Molecule {file_idx + 1}:')
        print('(non-aligned) shape scores:', [np.round(r, 3) for r in repeated_rocs])
        print('tanimoto chemical similarity:', [np.round(r, 3) for r in repeated_tanimoto])
        print()
        file_idx += 1
        
        reference_mols_list.append(reference_mols[0])
        unaligned_mols_list.append(unaligned_mols)
        
        if file_idx == total_evaluations:
            break

    molecules = [mol for mols in unaligned_mols_list for mol in mols]
    molecules_smiles = [Chem.MolToSmiles(mol) for mol in molecules]
    molecules_smiles = list(set(molecules_smiles))
    random.shuffle(molecules_smiles)
    return molecules_smiles
