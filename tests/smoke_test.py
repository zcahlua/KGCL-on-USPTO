"""Smoke test: instantiate KGCL model, run forward pass and predict on a toy SMILES."""
import sys
import traceback
import torch
from rdkit import Chem
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)

from kgcl_retro.models import KGCL
from kgcl_retro.chemistry.graphs import MolGraph, Vocab
from kgcl_retro.chemistry.features import ATOM_FDIM, BOND_FDIM
from kgcl_retro.data.collate import get_batch_graphs

# ── 1. Dummy vocabs ──────────────────────────────────────────────
atom_labels = [("ChangeBond", 1, "C"), ("ChangeBond", 2, "N"), ("Terminate",)]
bond_labels = [("ChangeBond", 0), ("ChangeBond", 1), ("ChangeBond", 2)]
atom_vocab = Vocab(atom_labels)
bond_vocab = Vocab(bond_labels)

# ── 2. Model config (small) ──────────────────────────────────────
config = {
    "n_atom_feat": ATOM_FDIM,
    "n_bond_feat": ATOM_FDIM + BOND_FDIM,
    "mpn_size": 64,
    "mlp_size": 64,
    "depth": 3,
    "dropout_mlp": 0.0,
    "dropout_mpn": 0.0,
    "atom_message": False,
    "use_attn": False,
    "n_heads": 4,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[smoke] Device: {device}")

model = KGCL(config=config, atom_vocab=atom_vocab, bond_vocab=bond_vocab, device=device)
model.to(device)
model.eval()
n_params = sum(p.nelement() for p in model.parameters()) / 1e6
print(f"[smoke] Model created – {n_params:.2f} M parameters")

# ── 3. Forward pass with dummy batch ─────────────────────────────
smi = "[CH3:1][OH:2]"  # methanol with atom maps
mol = Chem.MolFromSmiles(smi)
Chem.Kekulize(mol)
prod_graph = MolGraph(mol=Chem.Mol(mol), use_rxn_class=False)
prod_tensors, prod_scopes = get_batch_graphs([prod_graph], use_rxn_class=False)

# Wrap as a 1-step sequence (what forward() expects)
seq_input = [(prod_tensors, prod_scopes)]

with torch.no_grad():
    seq_edit_scores, batch_graph_outs = model(seq_input)

print(f"[smoke] Forward pass OK – edit_scores shape: {seq_edit_scores[0][0].shape}")
print(f"[smoke] Graph output shape: {batch_graph_outs[0].shape}")

# ── 4. Predict (autoregressive decoding) ─────────────────────────
with torch.no_grad():
    edits, edits_atom = model.predict(smi, max_steps=5)

print(f"[smoke] Predict OK – edits: {edits}")
print(f"[smoke] edits_atom: {edits_atom}")

print()
print("=" * 50)
print("  SMOKE TEST PASSED")
print("=" * 50)
