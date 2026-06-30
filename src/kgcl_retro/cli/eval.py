from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm

from kgcl_retro.data.dataset_config import get_dataset_spec, normalize_dataset_name
from kgcl_retro.models import BeamSearch, KGCL
from kgcl_retro.paths import resolve_project_paths

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RDLogger.logger().setLevel(RDLogger.CRITICAL)


def latest_checkpoint(root: Path, dataset: str, use_rxn_class: bool = False) -> Path:
    class_dir = "with_rxn_class" if use_rxn_class else "without_rxn_class"
    base = root / "experiments" / dataset / class_dir
    checkpoints = sorted(base.glob("*/epoch_*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {base}")
    return checkpoints[0]


def resolve_checkpoint(checkpoint: str, root: Path, dataset: str, use_rxn_class: bool = False) -> Path:
    if checkpoint == "auto":
        return latest_checkpoint(root, dataset, use_rxn_class)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    return checkpoint_path


def canonical_fragment_multiset(smiles: str) -> tuple[str, ...]:
    fragments: list[str] = []
    for fragment in str(smiles).split("."):
        if not fragment:
            continue
        mol = Chem.MolFromSmiles(fragment)
        if mol is not None:
            mol = Chem.RemoveHs(mol)
        if mol is None:
            fragments.append(fragment)
            continue
        for atom in mol.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True))
        if mol is None:
            fragments.append(fragment)
        else:
            fragments.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    return tuple(sorted(fragments))


def load_model(checkpoint_path: Path) -> KGCL:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if "saveables" not in checkpoint or "state" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} must contain 'state' and 'saveables'")
    model = KGCL(**checkpoint["saveables"], device=DEVICE)
    model.load_state_dict(checkpoint["state"])
    model.to(DEVICE)
    model.eval()
    return model


def output_path_for_arg(output: str, checkpoint_path: Path, root: Path) -> Path:
    output_path = Path(output)
    if output_path.is_absolute():
        return output_path
    if output == "pred_results.txt":
        return checkpoint_path.parent / output_path
    return root / output_path


def evaluate(args: argparse.Namespace) -> dict[int, float]:
    dataset = normalize_dataset_name(args.dataset)
    spec = get_dataset_spec(dataset)
    if args.forward:
        raise ValueError("This repository currently supports KGCL retrosynthesis, not forward reaction prediction.")
    if args.use_rxn_class and not spec.has_reaction_classes:
        raise ValueError(f"Dataset {dataset} does not provide reaction class labels; remove --use_rxn_class")
    if dataset == "uspto_50k":
        raise ValueError("Use kgcl-eval-50k for USPTO-50K evaluation to preserve existing behavior.")

    paths = resolve_project_paths(args.root_dir)
    checkpoint_path = resolve_checkpoint(args.checkpoint, paths.root, dataset, args.use_rxn_class)
    model = load_model(checkpoint_path)
    max_steps = args.max_steps if args.max_steps is not None else spec.max_edit_steps
    beam_model = BeamSearch(model=model, step_beam_size=10, beam_size=args.beam_size, use_rxn_class=args.use_rxn_class)

    test_file = paths.dataset_dir(dataset) / "test" / "test.file.kekulized"
    test_data = joblib.load(test_file)
    if not test_data:
        raise ValueError(f"No test reactions found in {test_file}")

    requested_top_k = [k for k in [1, 3, 5, 10] if k <= args.beam_size]
    top_k_hits = np.zeros(args.beam_size, dtype=float)
    edit_steps = []
    edit_steps_correct = []
    pred_file = output_path_for_arg(args.output, checkpoint_path, paths.root)
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    with pred_file.open("w") as handle:
        for idx in tqdm(range(len(test_data)), desc="Evaluating"):
            rxn_data = test_data[idx]
            reactants, product = rxn_data.rxn_smi.split(">>", 1)
            target = canonical_fragment_multiset(reactants)
            edit_steps.append(len(rxn_data.edits))
            rxn_class = rxn_data.rxn_class if args.use_rxn_class else None

            with torch.no_grad():
                top_k_results = beam_model.run_search(prod_smi=product, max_steps=max_steps, rxn_class=rxn_class)

            handle.write(f"({idx}) {rxn_data.rxn_smi}\n")
            matched = False
            for beam_idx, path in enumerate(top_k_results[: args.beam_size]):
                pred_smi = path.get("final_smi", "")
                pred = canonical_fragment_multiset(pred_smi)
                correct = pred == target
                edits = "|".join(
                    f"({edit};{prob})" for edit, prob in zip(path.get("rxn_actions", []), path.get("edits_prob", []))
                )
                handle.write(
                    f"{beam_idx} prediction_is_correct:{correct} probability:{path.get('prob')} {pred_smi} {edits}\n"
                )
                if correct and not matched:
                    top_k_hits[beam_idx] += 1
                    matched = True
            if matched:
                edit_steps_correct.append(len(rxn_data.edits))
            handle.write("\n")

        cumulative = np.cumsum(top_k_hits)
        metrics = {k: float(cumulative[k - 1] / len(test_data)) for k in requested_top_k}
        handle.write(f"top_k_accuracy:{metrics}\n")
        handle.write(f"edit_steps_reaction_number:{Counter(edit_steps)}\n")
        handle.write(f"edit_steps_reaction_prediction_correct:{Counter(edit_steps_correct)}\n")

    for k, value in metrics.items():
        print(f"top-{k}: {value:.4f}")
    print(f"predictions: {pred_file}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="General KGCL retrosynthesis evaluator")
    parser.add_argument("--dataset", default="uspto_mit")
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output", default="pred_results.txt")
    parser.add_argument("--use_rxn_class", action="store_true")
    parser.add_argument("--forward", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root_dir", default=".")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
