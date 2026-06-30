from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from rdkit import Chem


@dataclass
class AtomMappingReport:
    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    rdkit_parse_failures: int = 0
    unmapped_rows: int = 0
    product_atoms_missing_maps: int = 0
    product_maps_missing_from_reactants: int = 0
    reactant_only_mapped_atoms: int = 0
    skipped_rows: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _mol_maps(mol: Chem.Mol) -> list[int]:
    return [atom.GetAtomMapNum() for atom in mol.GetAtoms()]


def validate_atom_mapped_reaction(rxn_smi: str) -> list[str]:
    errors: list[str] = []
    if ">>" not in rxn_smi:
        return ["reaction must use retrosynthesis form reactants>>product"]
    reactants, product = rxn_smi.split(">>", 1)
    r_mol = Chem.MolFromSmiles(reactants)
    p_mol = Chem.MolFromSmiles(product)
    if r_mol is None or p_mol is None:
        return ["RDKit failed to parse reactants or product"]
    r_maps = _mol_maps(r_mol)
    p_maps = _mol_maps(p_mol)
    if not any(r_maps) and not any(p_maps):
        errors.append("reaction is completely unmapped")
    if any(m == 0 for m in p_maps):
        errors.append("one or more product atoms are missing atom-map numbers")
    missing_from_reactants = sorted({m for m in p_maps if m} - {m for m in r_maps if m})
    if missing_from_reactants:
        errors.append(f"product atom maps missing from reactants: {missing_from_reactants[:10]}")
    return errors


def update_report(report: AtomMappingReport, rxn_smi: str, errors: Iterable[str]) -> None:
    errors = list(errors)
    report.total_rows += 1
    if not errors:
        report.valid_rows += 1
        reactants, product = rxn_smi.split(">>", 1)
        r_mol = Chem.MolFromSmiles(reactants)
        p_mol = Chem.MolFromSmiles(product)
        r_only = {m for m in _mol_maps(r_mol) if m} - {m for m in _mol_maps(p_mol) if m}
        report.reactant_only_mapped_atoms += len(r_only)
        return
    report.invalid_rows += 1
    if any("parse" in e for e in errors):
        report.rdkit_parse_failures += 1
    if any("completely unmapped" in e for e in errors):
        report.unmapped_rows += 1
    if any("missing atom-map" in e for e in errors):
        report.product_atoms_missing_maps += 1
    if any("missing from reactants" in e for e in errors):
        report.product_maps_missing_from_reactants += 1
