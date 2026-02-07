import os
import sys
import csv
from typing import Dict, List, Optional

from rdkit import Chem
import warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "SQUID"))
from do_one import generate_molecules  # noqa: E102


def canon(s: str) -> Optional[str]:
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)


def pad(xs: List[str], n: int) -> List[str]:
    xs = xs[:n]
    while len(xs) < n:
        xs.append("")
    return xs


def to_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def sample(input_smiles: str, cache: Dict[str, List[str]]) -> List[str]:
    key = canon(input_smiles) or input_smiles
    if key in cache:
        return cache[key][:]

    seen = set()
    out: List[str] = []

    for s in to_list(
        generate_molecules(
            input_smiles,
            target_samples=10,
            n_conformers=5,
            max_seeds_per_conformer=10,
            reps_per_seed=10,
            max_total_attempts=100,
            seed_base=0,
            
        )
    ):
        cs = canon(s)
        if cs and cs not in seen:
            seen.add(cs)
            out.append(cs)
            if len(out) == 10:
                break

    cache[key] = pad(out, 10)
    return cache[key][:]


def main():
    if len(sys.argv) < 3:
        print("Usage: python run_batch.py input.csv output.csv", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    cache: Dict[str, List[str]] = {}

    smiles: List[str] = []
    with open(input_file, "r", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if row and row[0]:
                smiles.append(row[0].strip())

    header = [f"smi-{i:03d}" for i in range(10)]
    rows = [sample(s, cache) for s in smiles]

    with open(output_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    main()
