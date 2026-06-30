from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from kgcl_retro.chemistry.atom_mapping import AtomMappingReport, update_report, validate_atom_mapped_reaction
from kgcl_retro.data.dataset_config import csv_split_name, normalize_split_name
from kgcl_retro.paths import resolve_project_paths

RXN_COL = "reactants>reagents>production"
REACTION_COLUMNS = ["rxn_smiles", "reaction_smiles", "smiles", "reactants>reagents>product", RXN_COL]
SPLITS = ["train", "valid", "test"]


def untokenize(line: str) -> str:
    return "".join(str(line).strip().split())


def split_reaction(value: str, include_reagents: bool = False) -> tuple[str, str | None]:
    value = untokenize(value)
    if ">>" in value:
        reactants, product = value.split(">>", 1)
        return reactants, product
    parts = value.split(">")
    if len(parts) == 3:
        reactants, reagents, product = parts
        if include_reagents and reagents:
            reactants = ".".join(x for x in [reactants, reagents] if x)
        return reactants, product
    if len(parts) == 2:
        reactants, reagents = parts
        if include_reagents and reagents:
            reactants = ".".join(x for x in [reactants, reagents] if x)
        logging.warning("Source line has reactants>reagents but no product; using target product")
        return reactants, None
    logging.warning("Source line has no '>'; treating whole source as reactants and it may include reagents")
    return value, None


def build_rxn(reactants: str, product: str) -> str:
    return f"{untokenize(reactants)}>>{untokenize(product)}"


def _should_skip_mapping_error(errors: Iterable[str], allow_unmapped_skip: bool, skip_invalid: bool) -> bool:
    errors = list(errors)
    if any("RDKit failed" in error or "reaction must use" in error for error in errors):
        return skip_invalid
    return allow_unmapped_skip


def handle_row(
    rows: list[dict],
    split: str,
    row_id,
    rxn_smi: str,
    allow_unmapped_skip: bool,
    skip_invalid: bool,
    report: AtomMappingReport,
) -> None:
    errors = validate_atom_mapped_reaction(rxn_smi)
    update_report(report, rxn_smi, errors)
    if errors:
        msg = f"Invalid USPTO-MIT row in {split} row {row_id}: {'; '.join(errors)}; reaction={rxn_smi[:160]}"
        if _should_skip_mapping_error(errors, allow_unmapped_skip, skip_invalid):
            report.skipped_rows += 1
            logging.warning(msg + " -- skipping")
            return
        raise ValueError(msg)
    rows.append({"id": row_id, RXN_COL: rxn_smi})


def import_text(
    input_path: Path,
    split: str,
    include_reagents: bool,
    allow_unmapped_skip: bool,
    report: AtomMappingReport,
    skip_invalid: bool = False,
) -> list[dict]:
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
    for idx, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
        try:
            reactants, src_product = split_reaction(src_line, include_reagents)
            product = untokenize(tgt_line) or src_product
            if not product:
                raise ValueError("missing product")
            rxn_smi = build_rxn(reactants, product)
        except Exception as e:
            if skip_invalid:
                report.total_rows += 1
                report.invalid_rows += 1
                report.skipped_rows += 1
                logging.warning("Could not parse %s row %s: %s -- skipping", split, idx, e)
                continue
            raise ValueError(f"Could not parse {split} row {idx}: {e}") from e
        handle_row(rows, split, idx, rxn_smi, allow_unmapped_skip, skip_invalid, report)
    return rows


def _read_split_csv(input_path: Path, split: str) -> pd.DataFrame:
    if input_path.is_file():
        df = pd.read_csv(input_path)
        if "split" not in df.columns:
            raise ValueError("Single CSV input requires a 'split' column with train/val/valid/validation/test values")
        normalized = df["split"].map(lambda value: normalize_split_name(str(value)))
        return df.loc[normalized == normalize_split_name(split)].copy()

    csv_split = csv_split_name(split)
    path = input_path / f"{csv_split}.csv"
    if normalize_split_name(split) == "valid" and not path.exists():
        path = input_path / "valid.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV for split {split}: {path}")
    return pd.read_csv(path)


def _parse_csv_row(row: pd.Series, columns: list[str], split: str, row_id, include_reagents: bool) -> str:
    if "reactants" in columns and ("product" in columns or "production" in columns):
        product = row["product"] if "product" in columns else row["production"]
        reactants = str(row["reactants"])
        reagents = str(row.get("reagents", "")) if not pd.isna(row.get("reagents", "")) else ""
        if include_reagents and reagents:
            reactants = ".".join(x for x in [reactants, reagents] if x)
        return build_rxn(reactants, str(product))
    if "src" in columns and "tgt" in columns:
        reactants, src_product = split_reaction(row["src"], include_reagents)
        product = row["tgt"] if not pd.isna(row["tgt"]) else src_product
        if product is None:
            raise ValueError("missing tgt/product")
        return build_rxn(reactants, str(product))
    for col in REACTION_COLUMNS:
        if col in columns:
            reactants, product = split_reaction(row[col], include_reagents)
            if product is None:
                raise ValueError(f"could not parse product from column {col}")
            return build_rxn(reactants, product)
    raise ValueError(f"could not parse row columns={columns}")


def import_csv(
    input_path: Path,
    split: str,
    include_reagents: bool,
    allow_unmapped_skip: bool,
    report: AtomMappingReport,
    skip_invalid: bool = False,
) -> list[dict]:
    df = _read_split_csv(input_path, split)
    rows: list[dict] = []
    columns = list(df.columns)
    for idx, row in df.iterrows():
        row_id = row.get("id", idx)
        try:
            rxn_smi = _parse_csv_row(row, columns, split, row_id, include_reagents)
        except Exception as e:
            if skip_invalid:
                report.total_rows += 1
                report.invalid_rows += 1
                report.skipped_rows += 1
                logging.warning("Could not parse %s row %s: %s -- skipping", split, row_id, e)
                continue
            raise ValueError(f"Could not parse {split} row {row_id}: {e}") from e
        handle_row(rows, split, row_id, rxn_smi, allow_unmapped_skip, skip_invalid, report)
    return rows


def detect_format(input_path: Path) -> str:
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        return "csv"
    if input_path.is_dir() and any((input_path / f"src-{split}.txt").exists() for split in ["train", "val", "valid", "test"]):
        return "txt"
    if input_path.is_dir() and any((input_path / f"{split}.csv").exists() for split in ["train", "val", "valid", "test"]):
        return "csv"
    raise ValueError(f"Could not auto-detect USPTO-MIT input format in {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import USPTO-MIT/USPTO-480K in KGCL retrosynthesis form")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/uspto_mit"))
    parser.add_argument("--source-format", choices=["auto", "txt", "csv"], default="auto")
    parser.add_argument("--allow-unmapped-skip", action="store_true")
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--include-reagents-as-reactants", action="store_true")
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--root_dir", default=".")
    args = parser.parse_args()

    root = resolve_project_paths(args.root_dir).root
    output = args.output if args.output.is_absolute() else root / args.output
    fmt = detect_format(args.input) if args.source_format == "auto" else args.source_format
    output.mkdir(parents=True, exist_ok=True)
    report = AtomMappingReport()
    for split in SPLITS:
        if fmt == "txt":
            rows = import_text(args.input, split, args.include_reagents_as_reactants, args.allow_unmapped_skip, report, args.skip_invalid)
        else:
            rows = import_csv(args.input, split, args.include_reagents_as_reactants, args.allow_unmapped_skip, report, args.skip_invalid)
        with (output / f"raw_{csv_split_name(split)}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", RXN_COL])
            writer.writeheader()
            writer.writerows(rows)
    if args.report_json:
        args.report_json.write_text(json.dumps(report.to_dict(), indent=2))
    print(json.dumps(report.to_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
