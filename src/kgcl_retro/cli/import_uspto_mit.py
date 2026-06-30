from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import pandas as pd

from kgcl_retro.chemistry.atom_mapping import AtomMappingReport, update_report, validate_atom_mapped_reaction
from kgcl_retro.data.dataset_config import csv_split_name, normalize_split_name
from kgcl_retro.paths import resolve_project_paths

RXN_COL = "reactants>reagents>production"
REACTION_COLUMNS = ["rxn_smiles", "reaction_smiles", "smiles", "reactants>reagents>product", RXN_COL]


def untokenize(line: str) -> str:
    return "".join(line.strip().split())


def split_reaction(value: str, include_reagents: bool = False) -> tuple[str, str | None]:
    value = untokenize(str(value))
    if ">>" in value:
        r, p = value.split(">>", 1)
        return r, p
    parts = value.split(">")
    if len(parts) == 3:
        r, reagents, p = parts
        if include_reagents and reagents:
            r = ".".join(x for x in [r, reagents] if x)
        return r, p
    if len(parts) == 2:
        r, reagents = parts
        if include_reagents and reagents:
            r = ".".join(x for x in [r, reagents] if x)
        logging.warning("Source line has reactants>reagents but no product; using target product")
        return r, None
    logging.warning("Source line has no '>'; treating whole source as reactants and it may include reagents")
    return value, None


def build_rxn(reactants: str, product: str) -> str:
    return f"{untokenize(reactants)}>>{untokenize(product)}"


def handle_row(rows: list[dict], split: str, row_id, rxn_smi: str, allow_skip: bool, report: AtomMappingReport) -> None:
    errors = validate_atom_mapped_reaction(rxn_smi)
    update_report(report, rxn_smi, errors)
    if errors:
        msg = f"Invalid atom mapping in {split} row {row_id}: {'; '.join(errors)}; reaction={rxn_smi[:160]}"
        if allow_skip:
            report.skipped_rows += 1
            logging.warning(msg + " -- skipping")
            return
        raise ValueError(msg)
    rows.append({"id": row_id, RXN_COL: rxn_smi})


def import_text(input_path: Path, split: str, include_reagents: bool, allow_skip: bool, report: AtomMappingReport) -> list[dict]:
    csv_split = csv_split_name(split)
    src = input_path / f"src-{csv_split}.txt"
    tgt = input_path / f"tgt-{csv_split}.txt"
    if normalize_split_name(split) == "valid" and not src.exists():
        src = input_path / "src-valid.txt"
        tgt = input_path / "tgt-valid.txt"
    if not src.exists() or not tgt.exists():
        raise FileNotFoundError(f"Missing text files for split {split}: {src} / {tgt}")
    src_lines = src.read_text().splitlines()
    tgt_lines = tgt.read_text().splitlines()
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f"Mismatched line counts for {split}: {src} vs {tgt}")
    rows: list[dict] = []
    for idx, (s, t) in enumerate(zip(src_lines, tgt_lines)):
        reactants, src_product = split_reaction(s, include_reagents)
        product = untokenize(t) or src_product
        if not product:
            raise ValueError(f"Missing product for {split} row {idx}")
        handle_row(rows, split, idx, build_rxn(reactants, product), allow_skip, report)
    return rows


def import_csv(input_path: Path, split: str, include_reagents: bool, allow_skip: bool, report: AtomMappingReport) -> list[dict]:
    csv_split = csv_split_name(split)
    path = input_path / f"{csv_split}.csv" if input_path.is_dir() else input_path
    if normalize_split_name(split) == "valid" and input_path.is_dir() and not path.exists():
        path = input_path / "valid.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV for split {split}: {path}")
    df = pd.read_csv(path)
    rows: list[dict] = []
    for idx, row in df.iterrows():
        row_id = row.get("id", idx)
        rxn_smi = None
        if "reactants" in df.columns and ("product" in df.columns or "production" in df.columns):
            product = row.get("product", row.get("production"))
            reactants = str(row["reactants"])
            reagents = str(row.get("reagents", "")) if not pd.isna(row.get("reagents", "")) else ""
            if include_reagents and reagents:
                reactants = ".".join(x for x in [reactants, reagents] if x)
            rxn_smi = build_rxn(reactants, str(product))
        elif "src" in df.columns and "tgt" in df.columns:
            reactants, src_product = split_reaction(row["src"], include_reagents)
            rxn_smi = build_rxn(reactants, str(row["tgt"] or src_product))
        else:
            for col in REACTION_COLUMNS:
                if col in df.columns:
                    reactants, product = split_reaction(row[col], include_reagents)
                    if product is None:
                        raise ValueError(f"Could not parse product from {split} row {row_id} column {col}")
                    rxn_smi = build_rxn(reactants, product)
                    break
        if rxn_smi is None:
            raise ValueError(f"Could not parse {split} row {row_id}; columns={list(df.columns)}")
        handle_row(rows, split, row_id, rxn_smi, allow_skip, report)
    return rows


def detect_format(input_path: Path) -> str:
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        return "csv"
    if input_path.is_dir() and any((input_path / f"src-{s}.txt").exists() for s in ["train", "val", "valid", "test"]):
        return "txt"
    if input_path.is_dir() and any((input_path / f"{s}.csv").exists() for s in ["train", "val", "valid", "test"]):
        return "csv"
    raise ValueError(f"Could not auto-detect USPTO-MIT input format in {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import USPTO-MIT/USPTO-480K in KGCL retrosynthesis form",
        epilog=(
            "Primary atom-mapped USPTO-480K source:\n"
            "https://github.com/wengong-jin/nips17-rexgen/raw/master/USPTO/data.zip"
        ),
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/uspto_mit"))
    parser.add_argument("--source-format", choices=["auto", "txt", "csv"], default="auto")
    parser.add_argument("--allow-unmapped-skip", action="store_true")
    parser.add_argument("--include-reagents-as-reactants", action="store_true")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--root_dir", default=".")
    args = parser.parse_args()
    root = resolve_project_paths(args.root_dir).root
    output = args.output if args.output.is_absolute() else root / args.output
    fmt = detect_format(args.input) if args.source_format == "auto" else args.source_format
    output.mkdir(parents=True, exist_ok=True)
    report = AtomMappingReport()
    for split in ["train", "valid", "test"]:
        rows = import_text(args.input, split, args.include_reagents_as_reactants, args.allow_unmapped_skip, report) if fmt == "txt" else import_csv(args.input, split, args.include_reagents_as_reactants, args.allow_unmapped_skip, report)
        with (output / f"raw_{csv_split_name(split)}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", RXN_COL])
            writer.writeheader(); writer.writerows(rows)
    if args.report_json:
        args.report_json.write_text(json.dumps(report.to_dict(), indent=2))
    print(json.dumps(report.to_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
