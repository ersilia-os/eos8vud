import csv
import os
import random
import sys
import warnings
from typing import List

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "SQUID"))
from do_one import InputLimit, generate_molecules  # noqa: E402
from output_checks import OutputFilter  # noqa: E402

N_OUTPUTS = 10


class RowFailed(Exception):
    """A row that could not be filled.

    ``kind`` is "input" when the input is the cause (unusable input, or one the model
    cannot grow) and "model" otherwise. The run exits non-zero only if every row failed
    and one of them is a "model" failure.
    """

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def describe(exc):
    """One line for the log: the type and first message line of an error."""
    lines = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {lines[0] if lines else ''}"


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


def sample(input_smiles: str) -> List[str]:
    """Return up to N_OUTPUTS checked molecules for one input, padded with empty strings.

    Nothing is cached and every call draws a fresh random seed base, so the same input
    gives different molecules each time (also when it appears twice in one file).
    """
    checks = OutputFilter(input_smiles)
    if not checks.ok:
        raise RowFailed(
            "input", "input is empty, unparsable, multi-component or has no InChIKey"
        )

    try:
        generated = generate_molecules(
            input_smiles,
            target_samples=N_OUTPUTS,
            n_conformers=5,
            max_seeds_per_conformer=10,
            # 10 seeds x 2 reps = 20 attempts per conformer, so the budget of 100 attempts
            # rotates over the 5 conformers (10 reps used up the whole budget on the first)
            reps_per_seed=2,
            max_total_attempts=100,
            seed_base=random.SystemRandom().randrange(2**31),
            # only molecules that pass the checks count towards N_OUTPUTS
            accept=checks.accept,
        )
    except InputLimit as exc:
        raise RowFailed("input", str(exc)) from exc

    out = to_list(generated)[:N_OUTPUTS]
    print(f"[INFO] {input_smiles} {checks.summary()} returned={len(out)}")
    return pad(out, N_OUTPUTS)


def main():
    if len(sys.argv) < 3:
        print("Usage: python run_batch.py input.csv output.csv", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # one entry per input line, so a blank line gives an empty row instead of shifting the rest
    smiles: List[str] = []
    with open(input_file, "r", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            smiles.append(row[0].strip() if row else "")

    header = [f"smi_{i:01d}" for i in range(N_OUTPUTS)]
    rows = []
    failed = []  # (kind, message) of each row that could not be filled
    for s in smiles:
        try:
            rows.append(sample(s))
        except Exception as e:
            # anything that is not a RowFailed counts as the model failing
            kind = getattr(e, "kind", "model")
            message = str(e) if isinstance(e, RowFailed) else describe(e)
            print(f"[ERROR] {s!r}: {message}", file=sys.stderr)
            failed.append((kind, message))
            rows.append([""] * N_OUTPUTS)

    with open(output_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # a file of inputs the model cannot use is a valid (empty) answer; a broken model is not
    model_errors = [m for kind, m in failed if kind == "model"]
    if rows and len(failed) == len(rows) and model_errors:
        print(
            f"[ERROR] every input failed, first model error: {model_errors[0]}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
