from types import SimpleNamespace

import pytest

from kgcl_retro.chemistry.atom_mapping import AtomMappingReport, update_report, validate_atom_mapped_reaction
from kgcl_retro.cli.import_uspto_mit import RXN_COL, import_csv, import_text
from kgcl_retro.data.dataset_config import normalize_dataset_name, normalize_split_name

MAPPED = "[CH3:1][CH2:2][O:3].[CH3:4][Cl:5]>>[CH3:1][CH2:2][O:3][CH3:4]"


def test_dataset_aliases():
    for alias in ["uspto_mit", "uspto_480k", "USPTO-MIT", "USPTO-480K", "USPTO_MIT", "USPTO_480k", "mit", "480k"]:
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


def test_importer_csv_split_directory_modes(tmp_path):
    pd = pytest.importorskip("pandas")
    pd.DataFrame({"rxn_smiles": [MAPPED]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"reactants": ["[CH3:1][CH2:2][O:3]"], "reagents": ["[Na+:9]"], "product": ["[CH3:1][CH2:2][O:3]"]}).to_csv(tmp_path / "val.csv", index=False)
    pd.DataFrame({RXN_COL: [MAPPED]}).to_csv(tmp_path / "test.csv", index=False)
    report = AtomMappingReport()
    assert import_csv(tmp_path, "train", False, False, report)[0][RXN_COL] == MAPPED
    assert import_csv(tmp_path, "valid", False, False, report)[0][RXN_COL] == "[CH3:1][CH2:2][O:3]>>[CH3:1][CH2:2][O:3]"
    assert import_csv(tmp_path, "test", False, False, report)[0][RXN_COL] == MAPPED


def test_importer_single_csv_requires_split_and_preserves_order(tmp_path):
    pd = pytest.importorskip("pandas")
    csv_path = tmp_path / "all.csv"
    pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "split": ["train", "validation", "test"],
            "rxn_smiles": [MAPPED, "[CH3:1][CH2:2][O:3]>>[CH3:1][CH2:2][O:3]", MAPPED],
        }
    ).to_csv(csv_path, index=False)
    report = AtomMappingReport()
    assert [row["id"] for row in import_csv(csv_path, "valid", False, False, report)] == ["b"]
    no_split = tmp_path / "no_split.csv"
    pd.DataFrame({"rxn_smiles": [MAPPED]}).to_csv(no_split, index=False)
    with pytest.raises(ValueError, match="split"):
        import_csv(no_split, "train", False, False, AtomMappingReport())


def test_atom_mapping_validation_reports_unmapped_and_missing_product_maps():
    assert any("completely unmapped" in e for e in validate_atom_mapped_reaction("CCO>>CCO"))
    errors = validate_atom_mapped_reaction("[CH3:1][OH:2]>>[CH3:1]O")
    assert any("missing atom-map" in e for e in errors)
    report = AtomMappingReport()
    update_report(report, "CCO>>CCO", validate_atom_mapped_reaction("CCO>>CCO"))
    assert report.invalid_rows == 1
    assert report.unmapped_rows == 1


def test_importer_skip_controls_for_unmapped_and_invalid_rows(tmp_path):
    pd = pytest.importorskip("pandas")
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"rxn_smiles": ["CCO>>CCO", "not_a_smiles>>CCO"]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="completely unmapped"):
        import_csv(tmp_path, "train", False, False, AtomMappingReport())
    report = AtomMappingReport()
    with pytest.raises(ValueError, match="RDKit failed"):
        import_csv(tmp_path, "train", False, True, report)
    report = AtomMappingReport()
    rows = import_csv(tmp_path, "train", False, True, report, skip_invalid=True)
    assert rows == []
    assert report.skipped_rows == 2


def test_eval_helpers_use_multisets_and_explicit_checkpoint(tmp_path, monkeypatch):
    import kgcl_retro.cli.eval as eval_cli

    assert eval_cli.canonical_fragment_multiset("[CH4:1].[CH4:2]") == ("C", "C")
    checkpoint = tmp_path / "chosen_epoch_7.pt"
    checkpoint.write_bytes(b"checkpoint")
    test_dir = tmp_path / "data/uspto_mit/test"
    test_dir.mkdir(parents=True)
    (test_dir / "test.file.kekulized").write_bytes(b"unused")
    output = tmp_path / "predictions.txt"
    calls = {"checkpoint": None}

    class DummyBeam:
        def __init__(self, model, step_beam_size, beam_size, use_rxn_class):
            self.beam_size = beam_size

        def run_search(self, prod_smi, max_steps, rxn_class=None):
            return [{"final_smi": "[CH4:1].[CH4:2]", "prob": 1.0, "rxn_actions": [], "edits_prob": []}]

    monkeypatch.setattr(eval_cli, "load_model", lambda path: calls.update(checkpoint=path) or object())
    monkeypatch.setattr(eval_cli, "BeamSearch", DummyBeam)
    monkeypatch.setattr(eval_cli.joblib, "load", lambda path: [SimpleNamespace(rxn_smi="[CH4:1].[CH4:2]>>[CH3:1][CH3:2]", edits=["Terminate"], rxn_class=None)])
    args = SimpleNamespace(
        dataset="uspto_mit",
        checkpoint=str(checkpoint),
        beam_size=5,
        max_steps=3,
        output=str(output),
        use_rxn_class=False,
        forward=False,
        root_dir=str(tmp_path),
    )
    metrics = eval_cli.evaluate(args)
    assert calls["checkpoint"] == checkpoint
    assert metrics == {1: 1.0, 3: 1.0, 5: 1.0}
    assert "top_k_accuracy" in output.read_text()


def test_entry_points_import():
    import kgcl_retro.cli.import_uspto_mit as importer
    import kgcl_retro.cli.eval as eval_cli

    assert callable(importer.main)
    assert callable(eval_cli.main)


def test_tiny_pipeline_smoke(tmp_path):
    import subprocess

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for split in ["train", "val", "test"]:
        (input_dir / f"src-{split}.txt").write_text("[CH3:1][CH2:2][CH3:3]\n[CH3:1][CH2:2][CH3:3]\n")
        (input_dir / f"tgt-{split}.txt").write_text("[CH3:1][CH:2]=[CH2:3]\n[CH3:1][CH:2]=[CH2:3]\n")
    root = tmp_path / "root"
    cmds = [
        ["kgcl-import-uspto-mit", "--input", str(input_dir), "--output", "data/uspto_mit", "--source-format", "auto", "--root_dir", str(root)],
        ["kgcl-canonicalize", "--dataset", "uspto_mit", "--mode", "train", "--root_dir", str(root)],
        ["kgcl-preprocess", "--dataset", "uspto_mit", "--mode", "train", "--root_dir", str(root), "--lg_min_freq", "1"],
        ["kgcl-prepare-data", "--dataset", "uspto_mit", "--mode", "train", "--batch_size", "2", "--root_dir", str(root)],
    ]
    for cmd in cmds:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
    assert (root / "data/uspto_mit/canonicalized_train.csv").exists()
    assert (root / "data/uspto_mit/train/atom_lg_vocab.txt").exists()
    assert list((root / "data/uspto_mit/train/without_rxn_class").glob("batch-*.pt"))
