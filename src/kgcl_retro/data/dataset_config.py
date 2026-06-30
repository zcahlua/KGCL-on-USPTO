from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: tuple[str, ...]
    has_reaction_classes: bool
    default_lr: float
    max_edit_steps: int
    lg_min_freq: int
    round_trip_eval_default: bool = True


DATASET_SPECS: dict[str, DatasetSpec] = {
    "uspto_50k": DatasetSpec("uspto_50k", ("uspto_50k", "USPTO_50k", "USPTO-50K"), True, 1e-3, 9, 0, True),
    "uspto_full": DatasetSpec("uspto_full", ("uspto_full", "USPTO_full", "USPTO-FULL", "full"), False, 1e-4, 9, 50, True),
    "uspto_mit": DatasetSpec(
        "uspto_mit",
        ("uspto_mit", "uspto_480k", "USPTO-MIT", "USPTO-480K", "USPTO_MIT", "USPTO_480k", "mit", "480k"),
        False,
        1e-4,
        9,
        20,
        False,
    ),
}

_ALIAS_TO_NAME = {
    alias.lower().replace("-", "_"): name
    for name, spec in DATASET_SPECS.items()
    for alias in spec.aliases
}


def normalize_dataset_name(dataset: str) -> str:
    key = dataset.lower().replace("-", "_")
    if key not in _ALIAS_TO_NAME:
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported datasets: {', '.join(sorted(DATASET_SPECS))}")
    return _ALIAS_TO_NAME[key]


def get_dataset_spec(dataset: str) -> DatasetSpec:
    return DATASET_SPECS[normalize_dataset_name(dataset)]


def normalize_split_name(mode: str) -> str:
    key = mode.lower()
    if key == "train":
        return "train"
    if key in {"val", "valid", "validation"}:
        return "valid"
    if key == "test":
        return "test"
    raise ValueError("Unsupported split '%s'. Use train, val/valid/validation, or test." % mode)


def csv_split_name(mode: str) -> str:
    split = normalize_split_name(mode)
    return "val" if split == "valid" else split


def find_split_csv(dataset_dir: str | Path, prefix: str, mode: str) -> Path:
    dataset_dir = Path(dataset_dir)
    split = normalize_split_name(mode)
    candidates = [dataset_dir / f"{prefix}_{csv_split_name(split)}.csv"]
    if split == "valid":
        candidates.append(dataset_dir / f"{prefix}_valid.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {prefix} CSV for split '{mode}' in {dataset_dir}; tried {', '.join(map(str, candidates))}")


def output_split_csv(dataset_dir: str | Path, prefix: str, mode: str) -> Path:
    return Path(dataset_dir) / f"{prefix}_{csv_split_name(mode)}.csv"
