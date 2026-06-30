from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
GITIGNORE = ROOT / ".gitignore"


def test_readme_documents_uspto_480k_sources():
    text = README.read_text(encoding="utf-8")

    assert "https://github.com/wengong-jin/nips17-rexgen/tree/master/USPTO" in text
    assert "https://github.com/wengong-jin/nips17-rexgen/raw/master/USPTO/data.zip" in text
    assert "https://github.com/pschwllr/MolecularTransformer#pre-processing" in text


def test_readme_documents_atom_mapping_and_no_commit_warnings():
    text = README.read_text(encoding="utf-8").lower()

    assert "requires atom-mapped" in text
    assert "do not commit" in text
    assert "not be assumed to be directly kgcl-ready" in text


def test_gitignore_excludes_downloaded_uspto_data():
    patterns = set(GITIGNORE.read_text(encoding="utf-8").splitlines())

    assert "data/downloads/" in patterns
    assert "data/uspto_mit/" in patterns
