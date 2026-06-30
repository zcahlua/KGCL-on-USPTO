import csv
import json
from pathlib import Path

import pytest

from kgcl_retro.chemistry.atom_mapping import AtomMappingReport, update_report, validate_atom_mapped_reaction
from kgcl_retro.cli.import_uspto_mit import import_csv, import_text
from kgcl_retro.data.dataset_config import normalize_dataset_name, normalize_split_name

RXN_COL = "reactants>reagents>production"
MAPPED = "[CH3:1][CH2:2][O:3].[CH3:4][Cl:5]>>[CH3:1][CH2:2][O:3][CH3:4]"


def test_dataset_aliases():
    for alias in ["uspto_mit", "uspto_480k", "USPTO-MIT", "USPTO-480K", "USPTO_MIT", "USPTO_480k"]:
        assert normalize_dataset_name(alias) == "uspto_mit"


def test_split_aliases():
    assert normalize_split_name("val") == "valid"
    assert normalize_split_name("valid") == "valid"
    assert normalize_split_name("validation") == "valid"


def test_importer_text_mode_drops_reagents_and_untokenizes(tmp_path):
    for split in ["train", "val", "test"]:
        (tmp_path / f"src-{split}.txt").write_text("[CH3:1] [CH2:2] [O:3] > [Na+:9] > ignored\n")
        (tmp_path / f"tgt-{split}.txt").write_text("[CH3:1] [CH2:2] [O:3]\n")
    report = AtomMappingReport()
    rows = import_text(tmp_path, "valid", False, False, report)
    assert rows == [{"id": 0, RXN_COL: "[CH3:1][CH2:2][O:3]>>[CH3:1][CH2:2][O:3]"}]


def test_importer_csv_modes(tmp_path):
    pd = pytest.importorskip("pandas")
    pd.DataFrame({"rxn_smiles": [MAPPED]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"reactants": ["[CH3:1][CH2:2][O:3]"], "reagents": ["[Na+:9]"], "product": ["[CH3:1][CH2:2][O:3]"]}).to_csv(tmp_path / "val.csv", index=False)
    pd.DataFrame({RXN_COL: [MAPPED]}).to_csv(tmp_path / "test.csv", index=False)
    report = AtomMappingReport()
    assert import_csv(tmp_path, "train", False, False, report)[0][RXN_COL] == MAPPED
    assert import_csv(tmp_path, "valid", False, False, report)[0][RXN_COL] == "[CH3:1][CH2:2][O:3]>>[CH3:1][CH2:2][O:3]"
    assert import_csv(tmp_path, "test", False, False, report)[0][RXN_COL] == MAPPED


def test_atom_mapping_validation_reports_unmapped_and_missing_product_maps():
    assert any("completely unmapped" in e for e in validate_atom_mapped_reaction("CCO>>CCO"))
    errors = validate_atom_mapped_reaction("[CH3:1][OH:2]>>[CH3:1]O")
    assert any("missing atom-map" in e for e in errors)
    report = AtomMappingReport()
    update_report(report, "CCO>>CCO", validate_atom_mapped_reaction("CCO>>CCO"))
    assert report.invalid_rows == 1
    assert report.unmapped_rows == 1


def test_entry_points_import():
    import kgcl_retro.cli.import_uspto_mit as importer
    import kgcl_retro.cli.eval as eval_cli

    assert callable(importer.main)
    assert callable(eval_cli.main)


def test_tiny_pipeline_smoke(tmp_path):
    import subprocess, sys
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for split in ["train", "val", "test"]:
        (input_dir / f"src-{split}.txt").write_text("[CH3:1][CH2:2][CH3:3]\n[CH3:1][CH2:2][CH3:3]\n")
        (input_dir / f"tgt-{split}.txt").write_text("[CH3:1][CH:2]=[CH2:3]\n[CH3:1][CH:2]=[CH2:3]\n")
    root = tmp_path / "root"
    cmds = [
        ["kgcl-import-uspto-mit", "--input", str(input_dir), "--output", "data/uspto_mit", "--source-format", "auto", "--root_dir", str(root)],
        ["kgcl-canonicalize", "--dataset", "uspto_mit", "--mode", "train", "--root_dir", str(root)],
        ["kgcl-preprocess", "--dataset", "uspto_mit", "--mode", "train", "--root_dir", str(root)],
        ["kgcl-prepare-data", "--dataset", "uspto_mit", "--mode", "train", "--batch_size", "2", "--root_dir", str(root)],
    ]
    for cmd in cmds:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
    assert (root / "data/uspto_mit/canonicalized_train.csv").exists()
    assert (root / "data/uspto_mit/train/atom_lg_vocab.txt").exists()
    assert list((root / "data/uspto_mit/train/without_rxn_class").glob("batch-*.pt"))
