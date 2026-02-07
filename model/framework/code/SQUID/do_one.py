import os
import sys
import csv
import json
import random
import traceback
import time
from copy import deepcopy
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import torch

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem

from FPSim2 import FPSim2Engine

from models.vnn.models.vn_layers import *  # noqa: F401,F403
from models.vnn.models.utils.vn_dgcnn_util import get_graph_feature  # noqa: F401
from utils.general_utils import *  # noqa: F401,F403
from models.EGNN import *  # noqa: F401,F403
from models.models import *  # noqa: F401,F403

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

DEFAULT_TARGET_SAMPLES = 40
DEFAULT_N_CONFORMERS = 2
DEFAULT_MAX_SEEDS_PER_CONFORMER = 6
DEFAULT_REPS_PER_SEED = 4
DEFAULT_MAX_TOTAL_ATTEMPTS = 50
DEFAULT_TOTAL_EVALUATIONS = 50

ETKDG_MAX_ATTEMPTS = 200
MMFF_MAX_ITERS = 80
PRUNE_RMS = 0.2

interpolate_to_GNN_prior = 1.0
stop_threshold = 0.01
variational_GNN_factor = 1.0
ablateEqui = False


def t():
    return time.perf_counter()


def log(s):
    print(s, flush=True)


class Timer:
    def __init__(self, name):
        self.name = name
        self.t0 = None

    def __enter__(self):
        self.t0 = t()
        log(f"[TIMER] start {self.name}")
        return self

    def __exit__(self, *_):
        log(f"[TIMER] end   {self.name}  ({t() - self.t0:.3f}s)")


def norm_mol(m):
    if m is None:
        return None
    try:
        m = Chem.RemoveHs(m, sanitize=True)
    except Exception:
        m = Chem.RemoveHs(m, sanitize=False)
        try:
            Chem.SanitizeMol(m)
        except Exception:
            pass
    try:
        Chem.AssignStereochemistry(m, force=True, cleanIt=True)
    except Exception:
        pass
    return m


def canon(s):
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def seed_has_bond(m, seed):
    if m is None or not isinstance(seed, (list, tuple)) or len(seed) < 2:
        return False
    try:
        ss = set(int(i) for i in seed)
    except Exception:
        return False
    for b in m.GetBonds():
        a1 = b.GetBeginAtomIdx()
        a2 = b.GetEndAtomIdx()
        if a1 in ss and a2 in ss:
            return True
    return False


def fallback_seeds(m, max_seeds=64):
    if m is None:
        return []
    seeds = []
    seen = set()
    for b in m.GetBonds():
        u = b.GetBeginAtomIdx()
        v = b.GetEndAtomIdx()
        s = tuple(sorted((u, v)))
        if s not in seen:
            seeds.append(list(s))
            seen.add(s)
        if len(seeds) >= max_seeds:
            return seeds
    for v in range(m.GetNumAtoms()):
        nbrs = [n.GetIdx() for n in m.GetAtomWithIdx(v).GetNeighbors()]
        if len(nbrs) < 2:
            continue
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                u, w = nbrs[i], nbrs[j]
                s = tuple(sorted((u, v, w)))
                if s in seen:
                    continue
                if seed_has_bond(m, s):
                    seeds.append(list(s))
                    seen.add(s)
                if len(seeds) >= max_seeds:
                    return seeds
    return seeds


def prepare_molecule(smiles, k, seed=0, max_attempts=ETKDG_MAX_ATTEMPTS, max_iters=MMFF_MAX_ITERS, pick_best=False):
    log(f"[INFO] prepare_molecule smiles={smiles} n={k} pick_best={pick_best}")
    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        log("[ERROR] RDKit could not parse SMILES")
        return []
    try:
        Chem.SanitizeMol(mol0)
    except Exception as e:
        log(f"[WARN] initial sanitize failed ({type(e).__name__}: {e}); continuing")
    molH = Chem.AddHs(mol0)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 0
    params.maxAttempts = int(max_attempts)
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.pruneRmsThresh = float(PRUNE_RMS)

    with Timer("EmbedMultipleConfs"):
        conf_ids = list(AllChem.EmbedMultipleConfs(molH, numConfs=int(k), params=params))
    log(f"[INFO] EmbedMultipleConfs generated={len(conf_ids)}")
    if not conf_ids:
        return []

    mmff_results = None
    mmff_ok = False
    for variant in ("MMFF94s", "MMFF94"):
        try:
            with Timer(f"MMFFOptimizeMoleculeConfs {variant}"):
                mmff_results = AllChem.MMFFOptimizeMoleculeConfs(molH, mmffVariant=variant, maxIters=int(max_iters))
            if mmff_results and any(status == 0 for status, _ in mmff_results):
                mmff_ok = True
                log(f"[INFO] MMFF optimization done ({variant})")
                break
        except Exception as e:
            log(f"[WARN] MMFF {variant} failed ({type(e).__name__}: {e})")

    if not mmff_ok:
        log("[WARN] MMFF failed/unavailable; trying UFF")
        mmff_results = []
        with Timer("UFFOptimizeMolecule (loop)"):
            for cid in conf_ids:
                try:
                    status = AllChem.UFFOptimizeMolecule(molH, confId=int(cid), maxIters=int(max_iters))
                    mmff_results.append((status, float("inf")))
                except Exception:
                    mmff_results.append((1, float("inf")))

    chosen = conf_ids
    if pick_best and mmff_results is not None and len(mmff_results) == len(conf_ids):
        best_i = min(range(len(mmff_results)), key=lambda i: mmff_results[i][1])
        chosen = [conf_ids[best_i]]
        log(f"[INFO] selected best conformer idx={best_i} energy={mmff_results[best_i][1]:.4f}")

    out = []
    for cid in chosen:
        mH = Chem.Mol(molH)
        conf = Chem.Conformer(molH.GetConformer(int(cid)))
        mH.RemoveAllConformers()
        mH.AddConformer(conf, assignId=True)
        m = Chem.RemoveHs(mH)
        try:
            Chem.AssignAtomChiralTagsFromStructure(m, replaceExistingTags=True)
        except Exception:
            pass
        try:
            Chem.AssignStereochemistry(m, force=True, cleanIt=True)
        except Exception:
            pass
        out.append(m)
    log(f"[INFO] returning conformers={len(out)}")
    return out


def is_seedable(smiles):
    m = norm_mol(Chem.MolFromSmiles(smiles))
    if m is None:
        return False
    try:
        seeds = get_starting_seeds(m, AtomFragment_database, fragment_library_atom_features, unique_atoms, bond_lookup)
        seeds = [s for s in seeds if seed_has_bond(m, s)]
        return len(seeds) > 0
    except Exception as e:
        print(f"Does not have seed: {e}")
        return False


def seedable_smiles(smiles, max_scan=2000):
    if is_seedable(smiles):
        return smiles

    query = smiles
    with Timer("FPSim2 similarity"):
        results = fp0.similarity(query, search_cutoffs["atompair"], n_workers=1)
        if len(results) == 0:
            results = fp1.similarity(query, search_cutoffs["morgan"], n_workers=1)
        if len(results) == 0:
            results = fp0.similarity(query, 0.0, n_workers=1)

    log(f"[INFO] FPSim2 hits={len(results)}")
    if len(results) == 0:
        raise RuntimeError("No similarity search results returned; FPSim2 query failed or DB empty.")

    best_smiles = reference_smiles_list[results[0][0]]
    scanned = 0
    for ridx, _ in results:
        scanned += 1
        cand = reference_smiles_list[ridx]
        if is_seedable(cand):
            log(f"[INFO] replaced unseedable input with seedable reference after scanned={scanned}")
            return cand
        if scanned >= max_scan:
            break

    log(f"[WARN] no seedable found in first {max_scan}; using best_smiles")
    return best_smiles


def pick_seeds(mol, seed_base, m_idx, max_seeds_per_conformer):
    seeds = []
    try:
        with Timer("get_starting_seeds"):
            seeds = get_starting_seeds(mol, AtomFragment_database, fragment_library_atom_features, unique_atoms, bond_lookup)
    except Exception as e:
        log(f"[WARN] get_starting_seeds crashed ({type(e).__name__}: {e})")
        traceback.print_exc()

    seeds = [s for s in (seeds or []) if seed_has_bond(mol, s)]
    log(f"[INFO] seeds with >=1 bond after filter: {len(seeds)}")

    if len(seeds) == 0:
        log("[WARN] No valid seeds from get_starting_seeds; using fallback_connected_seeds")
        seeds = fallback_seeds(mol, max_seeds=max(32, int(max_seeds_per_conformer) * 8))
        log(f"[INFO] fallback seeds produced: {len(seeds)}")

    if len(seeds) == 0:
        return []

    rng = random.Random(seed_base + 9999 * (m_idx + 1))
    rng.shuffle(seeds)
    return seeds[: int(max_seeds_per_conformer)]


def center_conformer(m):
    xyz = np.array(m.GetConformer().GetPositions())
    com = np.mean(xyz, axis=0)
    xyz = xyz - com
    for i in range(m.GetNumAtoms()):
        x, y, z = xyz[i]
        m.GetConformer().SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return m


def build_seed_state(mol, mol_target, seed_obj, frame_generation, start):
    if len(frame_generation.iloc[0].partial_graph_indices) == 1:
        terminal = frame_generation.iloc[0:start].reset_index(drop=True)
        sequence = get_ground_truth_generation_sequence(terminal, AtomFragment_database, fragment_library_atom_features)
        mol0 = norm_mol(deepcopy(terminal.iloc[0].rdkit_mol_cistrans_stereo))
        if mol0 is None:
            raise RuntimeError("mol0 invalid")

        partial = deepcopy(terminal.iloc[0].partial_graph_indices_sorted)
        final_partial = deepcopy(terminal.iloc[-1].partial_graph_indices_sorted)

        ring_frags = get_ring_fragments(mol0)
        add = [list(f) for p in final_partial for f in ring_frags if p in f]
        add = [i for sub in add for i in sub]
        final_partial = list(set(final_partial).union(add))

        queue_indices = deepcopy(terminal.iloc[0].focal_indices_sorted)

        _, seed_mol, queue, positioned, atom_to_lib, _, _, _ = generate_seed_from_sequence(
            sequence,
            mol0,
            partial,
            queue_indices,
            AtomFragment_database,
            unique_atoms,
            bond_lookup,
            stop_after_sequence=True,
        )

        seed_nf = getNodeFeatures(seed_mol.GetAtoms())
        for k in atom_to_lib:
            seed_nf[k] = AtomFragment_database.iloc[atom_to_lib[k]].atom_features

        G = get_substructure_graph(mol0, final_partial)
        Gs = get_substructure_graph(seed_mol, list(range(seed_mol.GetNumAtoms())), node_features=seed_nf)
        nm = nx.algorithms.isomorphism.generic_node_match(["atom_features"], [None], [np.allclose])
        em = nx.algorithms.isomorphism.numerical_edge_match("bond_type", 1.0)
        GM = nx.algorithms.isomorphism.GraphMatcher(G, Gs, node_match=nm, edge_match=em)
        if not GM.is_isomorphic():
            raise RuntimeError("Seed graph not isomorphic to target substructure")
        idx_map = GM.mapping
        final_partial_indices = final_partial
        atom_to_library_ID_map = atom_to_lib
        return sequence, seed_mol, queue, positioned, atom_to_library_ID_map, idx_map, final_partial_indices

    partial = deepcopy(frame_generation.iloc[0].partial_graph_indices_sorted)
    frag_smiles = get_fragment_smiles(deepcopy(mol), partial)
    seed_mol = norm_mol(generate_conformer(frag_smiles))
    if seed_mol is None:
        raise RuntimeError("seed_mol invalid")
    idx_map = get_reindexing_map(deepcopy(mol), partial, seed_mol)
    positioned = sorted([idx_map[f] for f in partial])
    return [], seed_mol, [0], positioned, {}, idx_map, partial


def transfer_coords(mol, seed_mol, idx_map, final_partial_indices):
    if seed_mol.GetNumConformers() == 0:
        raise RuntimeError("seed_mol has no conformers")
    for i in final_partial_indices:
        x, y, z = mol.GetConformer().GetPositions()[i]
        seed_mol.GetConformer().SetAtomPosition(idx_map[i], Point3D(float(x), float(y), float(z)))


def run_growth(sequence, seed_mol, mol_target, positioned, queue, atom_to_lib):
    _, updated_mol, _, _, _, _, _, _, _, _ = generate_3D_mol_from_sequence(
        sequence=sequence,
        partial_mol=deepcopy(seed_mol),
        mol=deepcopy(mol_target),
        positioned_atoms_indices=deepcopy(positioned),
        queue=deepcopy(queue),
        atom_to_library_ID_map=deepcopy(atom_to_lib),
        model=model_3D,
        rocs_model=rocs_model_3D,
        AtomFragment_database=AtomFragment_database,
        unique_atoms=unique_atoms,
        bond_lookup=bond_lookup,
        N_points=5,
        N_points_rocs=5,
        stop_after_sequence=False,
        mask_first_stop=False,
        stochastic=False,
        chirality_scoring=True,
        stop_threshold=stop_threshold,
        steric_mask=True,
        variational_factor_equi=0.0,
        variational_factor_inv=0.0,
        interpolate_to_prior_equi=0.0,
        interpolate_to_prior_inv=0.0,
        use_variational_GNN=True,
        variational_GNN_factor=variational_GNN_factor,
        interpolate_to_GNN_prior=interpolate_to_GNN_prior,
        rocs_use_variational_GNN=False,
        rocs_variational_GNN_factor=0.0,
        rocs_interpolate_to_GNN_prior=0.0,
        pointCloudVar=pointCloudVar,
        rocs_pointCloudVar=rocs_pointCloudVar,
    )
    return updated_mol


def generate_molecules(
    smiles,
    device=torch.device("cpu"),
    target_samples=DEFAULT_TARGET_SAMPLES,
    n_conformers=DEFAULT_N_CONFORMERS,
    max_seeds_per_conformer=DEFAULT_MAX_SEEDS_PER_CONFORMER,
    reps_per_seed=DEFAULT_REPS_PER_SEED,
    max_total_attempts=DEFAULT_MAX_TOTAL_ATTEMPTS,
    seed_base=0,
    total_evaluations=DEFAULT_TOTAL_EVALUATIONS,
    debug_every=10,
):
    log(f"[INFO] generate_molecules input smiles={smiles}")

    if Chem.MolFromSmiles(smiles) is None:
        log("[WARN] RDKit could not parse input; trying FPSim2 replacement")
        with Timer("get_seedable_smiles"):
            smiles = seedable_smiles(smiles, max_scan=2000)
        log(f"[INFO] using smiles={smiles}")
    else:
        log("[INFO] RDKit parsed input OK; skipping FPSim2 replacement")

    mol2d = norm_mol(Chem.MolFromSmiles(smiles))
    if mol2d is None:
        log("[ERROR] could not create mol2d after parse")
        return []

    with Timer("prepare_molecule"):
        mols = prepare_molecule(smiles, int(n_conformers), seed=seed_base, pick_best=False)
    if len(mols) == 0:
        log("[ERROR] No 3D conformers could be prepared.")
        return []

    seen = set()
    unique = []
    total_attempts = 0
    total_failures = 0
    last_error = None
    file_idx = 0

    for m_idx, m0 in enumerate(mols):
        if len(unique) >= target_samples:
            break
        log(f"[INFO] ===== Conformer {m_idx + 1}/{len(mols)} =====")

        mol = norm_mol(deepcopy(m0))
        if mol is None or mol.GetNumConformers() == 0:
            continue

        mol = center_conformer(mol)
        mol_target = deepcopy(mol)

        seeds = pick_seeds(mol_target, seed_base, m_idx, max_seeds_per_conformer)
        if len(seeds) == 0:
            log("[WARN] No seeds; skipping")
            continue

        log(f"[INFO] using seeds={len(seeds)} reps_per_seed={reps_per_seed}")

        for seed_i, seed_obj in enumerate(seeds):
            for rep in range(int(reps_per_seed)):
                if len(unique) >= target_samples or total_attempts >= max_total_attempts:
                    break

                total_attempts += 1
                attempt_seed = int(seed_base + 100000 * (m_idx + 1) + 1000 * (seed_i + 1) + rep)
                random.seed(attempt_seed)
                np.random.seed(attempt_seed)
                torch.manual_seed(attempt_seed)

                try:
                    with torch.inference_mode():
                        frame_generation, frame_rocs = get_frame_terminalSeeds(
                            deepcopy(mol), seed_obj, AtomFragment_database, include_rocs=True
                        )
                        if frame_rocs is None or len(frame_rocs) == 0 or len(frame_generation) == 0:
                            raise RuntimeError("frame_generation or frame_rocs empty")

                        positions = list(frame_rocs.iloc[0].positions_before)
                        start = 0
                        for i in range(len(frame_generation)):
                            if (set(frame_generation.iloc[i].partial_graph_indices) == set(positions)) and (
                                frame_generation.iloc[i].next_atom_index == -1
                            ):
                                start = i + 1
                                break

                        sequence, seed_mol, queue, positioned, atom_to_lib, idx_map, final_partial = build_seed_state(
                            mol, mol_target, seed_obj, frame_generation, start
                        )

                        transfer_coords(mol, seed_mol, idx_map, final_partial)

                        updated_mol = run_growth(sequence, seed_mol, mol_target, positioned, queue, atom_to_lib)
                        if updated_mol is None or updated_mol.GetNumConformers() == 0:
                            raise RuntimeError("updated_mol invalid")

                        smi = canon(Chem.MolToSmiles(updated_mol))
                        if smi and smi not in seen:
                            seen.add(smi)
                            unique.append(smi)
                            log(f"[OK] sample {len(unique)}/{target_samples}: {smi}")

                except Exception as e:
                    total_failures += 1
                    last_error = e
                    if total_failures <= 3 or (debug_every > 0 and total_failures % debug_every == 0):
                        log(f"[ERROR] attempt failed (failures={total_failures}, attempts={total_attempts}) {type(e).__name__}: {e}")
                        traceback.print_exc()

        file_idx += 1
        if file_idx >= total_evaluations:
            break

    log("[INFO] ===== Summary =====")
    log(f"[INFO] total_attempts={total_attempts}")
    log(f"[INFO] total_failures={total_failures}")
    log(f"[INFO] unique_smiles={len(unique)}")
    if len(unique) == 0 and last_error is not None:
        log(f"[ERROR] last_error={type(last_error).__name__}: {last_error}")

    return unique


root = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(root, "..", "..", "..", "checkpoints")

log(f"[INFO] root={root}")
log(f"[INFO] CHECKPOINTS_DIR={CHECKPOINTS_DIR}")
log(f"[INFO] python={sys.executable}")
log(f"[INFO] torch={torch.__version__}")
log(f"[INFO] numpy={np.__version__}")
log(f"[INFO] rdkit={rdkit.__version__ if hasattr(rdkit, '__version__') else 'unknown'}")

import warnings

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=getattr(np, "VisibleDeprecationWarning", DeprecationWarning))

atomfrag_path = os.path.join(CHECKPOINTS_DIR, "data/MOSES2/MOSES2_training_val_AtomFragment_database.pkl")
bond_lookup_path = os.path.join(CHECKPOINTS_DIR, "data/MOSES2/MOSES2_training_val_bond_lookup.pkl")
unique_atoms_path = os.path.join(CHECKPOINTS_DIR, "data/MOSES2/MOSES2_training_val_unique_atoms.npy")

if not os.path.exists(atomfrag_path):
    raise FileNotFoundError(f"Missing AtomFragment_database: {atomfrag_path}")
if not os.path.exists(bond_lookup_path):
    raise FileNotFoundError(f"Missing bond_lookup: {bond_lookup_path}")
if not os.path.exists(unique_atoms_path):
    raise FileNotFoundError(f"Missing unique_atoms: {unique_atoms_path}")

with Timer("load fragment libraries"):
    AtomFragment_database = pd.read_pickle(atomfrag_path)
    AtomFragment_database = AtomFragment_database.iloc[1:].reset_index(drop=True)
    fragment_library_atom_features = np.concatenate(AtomFragment_database["atom_features"], axis=0).reshape((len(AtomFragment_database), -1))
    bond_lookup = pd.read_pickle(bond_lookup_path)
    unique_atoms = np.load(unique_atoms_path)

if not ablateEqui:
    model_3D_PATH = os.path.join(root, "trained_models/graph_generator.pt")
    rocs_model_3D_PATH = os.path.join(root, "trained_models/scorer.pt")
else:
    model_3D_PATH = os.path.join(root, "trained_models/graph_generator_ablateEqui.pt")
    rocs_model_3D_PATH = os.path.join(root, "trained_models/scorer_ablateEqui.pt")

if not os.path.exists(model_3D_PATH):
    raise FileNotFoundError(f"Missing model checkpoint: {model_3D_PATH}")
if not os.path.exists(rocs_model_3D_PATH):
    raise FileNotFoundError(f"Missing ROCS checkpoint: {rocs_model_3D_PATH}")

fp0_filename = os.path.join(CHECKPOINTS_DIR, "atompair_fps.h5")
fp1_filename = os.path.join(CHECKPOINTS_DIR, "morgan_fps.h5")
if not os.path.exists(fp0_filename):
    raise FileNotFoundError(f"Missing FPSim2 DB: {fp0_filename}")
if not os.path.exists(fp1_filename):
    raise FileNotFoundError(f"Missing FPSim2 DB: {fp1_filename}")

with Timer("load FPSim2 DBs"):
    fp0 = FPSim2Engine(fp0_filename)
    fp1 = FPSim2Engine(fp1_filename)

cutoff_path = os.path.join(CHECKPOINTS_DIR, "reasonable_fp_cutoffs.json")
ref_path = os.path.join(CHECKPOINTS_DIR, "reference_smiles.txt")
if not os.path.exists(cutoff_path):
    raise FileNotFoundError(f"Missing: {cutoff_path}")
if not os.path.exists(ref_path):
    raise FileNotFoundError(f"Missing: {ref_path}")

with Timer("load similarity cutoffs + reference smiles"):
    with open(cutoff_path, "r") as f:
        search_cutoffs = json.load(f)
    reference_smiles_list = []
    with open(ref_path, "r") as f:
        for r in csv.reader(f):
            if r and r[0]:
                reference_smiles_list.append(r[0])

pointCloudVar = 1.0 / (12.0 * 1.7)
rocs_pointCloudVar = 1.0 / (12.0 * 1.7)

with Timer("instantiate + load torch models"):
    model_3D = Model_Point_Cloud_Switched(
        input_nf=45,
        edges_in_d=5,
        n_knn=5,
        conv_dims=[32, 32, 64, 128],
        num_components=64,
        fragment_library_dim=64,
        N_fragment_layers=3,
        append_noise=False,
        N_members=125 - 1,
        EGNN_layer_dim=64,
        N_EGNN_layers=3,
        output_MLP_hidden_dim=64,
        pooling_MLP=False,
        shared_encoders=False,
        subtract_latent_space=True,
        variational=False,
        variational_mode="inv",
        variational_GNN=True,
        mix_node_inv_to_equi=True,
        mix_shape_to_nodes=True,
        ablate_HvarCat=False,
        predict_pairwise_properties=False,
        predict_mol_property=False,
        ablateEqui=ablateEqui,
        old_EGNN=False,
    ).float()

    rocs_model_3D = ROCS_Model_Point_Cloud(
        input_nf=45,
        edges_in_d=5,
        n_knn=10,
        conv_dims=[32, 32, 64, 128],
        num_components=64,
        fragment_library_dim=64,
        N_fragment_layers=3,
        append_noise=False,
        N_members=125 - 1,
        EGNN_layer_dim=64,
        N_EGNN_layers=3,
        output_MLP_hidden_dim=64,
        pooling_MLP=False,
        shared_encoders=False,
        subtract_latent_space=True,
        variational=False,
        variational_mode="inv",
        variational_GNN=False,
        mix_node_inv_to_equi=True,
        mix_shape_to_nodes=True,
        ablate_HvarCat=False,
        ablateEqui=ablateEqui,
        old_EGNN=False,
    ).float()

    model_3D.load_state_dict(torch.load(model_3D_PATH, map_location="cpu"), strict=True)
    rocs_model_3D.load_state_dict(torch.load(rocs_model_3D_PATH, map_location="cpu"), strict=True)
    model_3D.eval()
    rocs_model_3D.eval()
