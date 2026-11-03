"""Validity, echo and duplicate checks for generated molecules.

An output is kept only if it parses and sanitises, is a single component, has an
InChIKey, is not the input (compared on the first InChIKey block, so any
stereoisomer of the input counts as the input) and is not a repeat of an earlier
output (compared on the first InChIKey block too, so stereoisomers of one output
count as repeats). Written for Python 3.8 and RDKit 2022.09 or later.
"""

from collections import Counter

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")

REASONS = ("invalid", "multi_component", "no_inchikey", "echo", "duplicate", "formula")


def parse_single_component(smiles):
    """Return the RDKit molecule of a SMILES string, or None.

    Parameters
    ----------
    smiles : str
        Candidate SMILES. Anything that is not a non-empty string gives None.

    Returns
    -------
    rdkit.Chem.Mol or None
        The sanitised molecule, or None if it is empty, unparsable or has more
        than one component.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    return mol


def inchikey(mol):
    """Return the InChIKey of a molecule, or an empty string if InChI fails."""
    try:
        return MolToInchiKey(mol) or ""
    except Exception:
        return ""


class OutputFilter:
    """Filter the outputs generated for one input compound.

    Parameters
    ----------
    input_smiles : str
        The input compound. If it is empty, unparsable, multi-component or has
        no InChIKey, ``ok`` is False and every output is rejected.
    check_formula : bool, optional
        Also require each output to have the input's molecular formula.

    Attributes
    ----------
    ok : bool
        Whether the input can be used.
    formula : str or None
        Molecular formula of the input (only set when ``check_formula`` is True).
    rejected : collections.Counter
        Number of rejected outputs per reason (see ``REASONS``).
    kept : int
        Number of accepted outputs.
    """

    def __init__(self, input_smiles, check_formula=False):
        mol = parse_single_component(input_smiles)
        key = inchikey(mol) if mol is not None else ""
        self.ok = bool(key)
        self._echo_block = key.split("-")[0]
        self.formula = None
        if self.ok and check_formula:
            self.formula = rdMolDescriptors.CalcMolFormula(mol)
        self._seen = set()
        self.rejected = Counter()
        self.kept = 0

    def accept(self, smiles):
        """Return the canonical SMILES of a generated molecule, or None if rejected."""
        if not self.ok:
            self.rejected["invalid"] += 1
            return None
        if not isinstance(smiles, str) or not smiles.strip():
            self.rejected["invalid"] += 1
            return None
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
        except Exception:
            mol = None
        if mol is None or mol.GetNumAtoms() == 0:
            self.rejected["invalid"] += 1
            return None
        if len(Chem.GetMolFrags(mol)) != 1:
            self.rejected["multi_component"] += 1
            return None
        key = inchikey(mol)
        if not key:
            self.rejected["no_inchikey"] += 1
            return None
        if key.split("-")[0] == self._echo_block:
            self.rejected["echo"] += 1
            return None
        if (
            self.formula is not None
            and rdMolDescriptors.CalcMolFormula(mol) != self.formula
        ):
            self.rejected["formula"] += 1
            return None
        block = key.split("-")[0]
        if block in self._seen:
            self.rejected["duplicate"] += 1
            return None
        self._seen.add(block)
        self.kept += 1
        return Chem.MolToSmiles(mol)

    def summary(self):
        """One-line count of kept and rejected outputs, for the run log."""
        parts = ["kept=%d" % self.kept]
        parts += ["%s=%d" % (r, self.rejected[r]) for r in REASONS if self.rejected[r]]
        return " ".join(parts)
