from __future__ import annotations

import argparse
from pathlib import Path

from kgcl_retro.data.dataset_config import get_dataset_spec, normalize_dataset_name


def latest_checkpoint(root: Path, dataset: str, use_rxn_class: bool) -> Path:
    class_dir = "with_rxn_class" if use_rxn_class else "without_rxn_class"
    base = root / "experiments" / dataset / class_dir
    checkpoints = sorted(base.glob("*/epoch_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {base}")
    return checkpoints[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="General KGCL retrosynthesis evaluator")
    parser.add_argument("--dataset", default="uspto_mit")
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output", default="pred_results.txt")
    parser.add_argument("--use_rxn_class", action="store_true")
    parser.add_argument("--forward", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root_dir", default=".")
    args = parser.parse_args()
    dataset = normalize_dataset_name(args.dataset)
    spec = get_dataset_spec(dataset)
    if args.forward:
        raise ValueError("This repository currently supports KGCL retrosynthesis, not forward reaction prediction.")
    if args.use_rxn_class and not spec.has_reaction_classes:
        raise ValueError(f"Dataset {dataset} does not provide reaction class labels; remove --use_rxn_class")
    checkpoint = latest_checkpoint(Path(args.root_dir).resolve(), dataset, args.use_rxn_class) if args.checkpoint == "auto" else Path(args.checkpoint)
    # Delegate stable legacy implementations for existing datasets. USPTO-MIT shares the USPTO-full no-class path.
    if dataset not in {"uspto_mit", "uspto_full"}:
        raise ValueError("Use kgcl-eval-50k for reaction-class USPTO-50K evaluation.")
    from kgcl_retro.cli import eval_full
    import sys
    exp_dir = checkpoint.parent.name
    sys.argv = ["kgcl-eval-full", "--dataset", dataset, "--experiments", exp_dir, "--beam_size", str(args.beam_size), "--max_steps", str(args.max_steps or spec.max_edit_steps), "--root_dir", args.root_dir]
    eval_full.main()


if __name__ == "__main__":
    main()
