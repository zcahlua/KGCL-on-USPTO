# KGCL
## Title
KGCL: Knowledge-Enhanced Graph Contrastive Learning for Retrosynthesis Prediction Based on Molecular Graph Editing
## Environment Requirements  
- python = 3.11.8
- pytorch = 2.2.2
- numpy = 1.26.4
- rdkit = 2024.03.4

## Install as a Package

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

If RDKit is not available through pip in your environment, install RDKit with conda first:

```bash
conda install -c conda-forge rdkit
python -m pip install -e ".[dev]"
```

## Package CLI

The legacy scripts still work, but the package commands are preferred:

```bash
kgcl-canonicalize --dataset uspto_50k --mode train
kgcl-preprocess --dataset uspto_50k --mode train
kgcl-prepare-data --dataset uspto_50k --mode train
kgcl-train --dataset uspto_50k
kgcl-eval-50k --dataset uspto_50k
kgcl-eval-full --dataset uspto_full
kgcl-eval-roundtrip --dataset uspto_50k
```

## Data
The original datasets used in this paper are from:

USPTO-50K: [https://github.com/Hanjun-Dai/GLN](https://github.com/Hanjun-Dai/GLN) (schneider50k)

USPTO-FULL:[https://github.com/Hanjun-Dai/GLN](https://github.com/Hanjun-Dai/GLN) (uspto_multi)

The raw data, processed data can be accessed via [link](https://drive.google.com/drive/folders/11YMNrm7St-GgVF278orHSXk-EKM3ltqH?usp=sharing). The directory structure should be as follows:

```
KGCL
├───data
|   ├───uspto_50K
|   │       ├───canonicalized_test.csv
|   │       ├───canonicalized_train.csv
|   │       ├───canonicalized_val.csv
|   │       ├───raw_test.csv
|   │       ├───raw_train.csv
|   │       └───raw_val.csv
|   │       
|   │       
|   └───uspto_full
|           ├───canonicalized_test.csv
|           ├───canonicalized_train.csv
|           ├───canonicalized_val.csv
|           ├───raw_test.csv
|           ├───raw_train.csv
|           └───raw_val.csv
```
- Data
    - The raw data of the USPTO-50K dataset and USPTO-FULL dataset is stored in the corresponding folders in the files `raw_train.csv`, `raw_val.csv`, and `raw_test.csv`.
    - All the processed data are named `canonicalized_train.csv` , `canonicalized_val.csv` and `canonicalized_test.csv` and are put in the corresponding folders respectively.

## Data preprocessing
- generate the edit labels and the edits sequence for reaction 
```
python preprocess.py --mode train --dataset USPTO_50k \
python preprocess.py --mode valid --dataset USPTO_50k \
python preprocess.py --mode test --dataset USPTO_50k \ 
or
python preprocess.py --mode train --dataset uspto_full \
python preprocess.py --mode valid --dataset uspto_full \
python preprocess.py --mode test --dataset uspto_full \ 
```
-   Prepare the data for training without using reaction classes as a condition
```
python prepare_data.py --dataset USPTO_50k or uspto_full
```
- Prepare the data for training using reaction classes as a condition
```
python prepare_data.py --dataset USPTO_50k --use_rxn_class
```
## Train KGCL model

- Run the following to train the model with specified dataset without using reaction classes as a condition
```
python train.py --dataset uspto_50k or uspto_full
```
The trained model will be saved at KGCL/experiments/uspto_50k/without_rxn_class/

- Run the following to train the model with USPTO-50K dataset using reaction classes as a condition
```
python train.py --dataset uspto_50k --use_rxn_class
```
The trained model will be saved at KGCL/experiments/uspto_50k/with_rxn_class/
# Test
To evaluate the trained model, run
```
python eval.py or
python eval.py --use_rxn_class
```
The raw prediction file saved at KGCL/experiments/.../pred_results.txt
## Reproducing our results
- To reproduce our exact accuracy and MaxFrag accuracy results on USPTO-50K dataset, run

```
python eval.py --dataset uspto_50k \
python eval.py --dataset uspto_50k --use_rxn_class \
```
This will display the exact accuracy and MaxFrag accuracy results for reaction class unknown and known setting
- To reproduce our round-trip accuracy results on USPTO-50K dataset, run
```
python eval-rtacc.py
```
This will display the round-trip accuracy results for reaction class unknown setting
- To reproduce our exact accuracy results on USPTO-FULL dataset, run
```
python eval-full.py
```
This will display the exact accuracy results for reaction class unknown setting


## Downloading USPTO-MIT / USPTO-480K

KGCL requires atom-mapped reaction SMILES for graph-edit label extraction. For USPTO-MIT / USPTO-480K, the recommended starting point is the atom-mapped USPTO archive from the `wengong-jin/nips17-rexgen` repository:

- Source page: https://github.com/wengong-jin/nips17-rexgen/tree/master/USPTO
- Direct download: https://github.com/wengong-jin/nips17-rexgen/raw/master/USPTO/data.zip

That repository describes `USPTO/data.zip` as a train/dev/test split containing approximately 480K fully atom-mapped reactions. KGCL uses this dataset only in retrosynthesis-by-inversion mode: USPTO-MIT / USPTO-480K is commonly used for forward reaction prediction, but this repository reads each atom-mapped reaction as `reactants>>product`, uses the product graph as model input, and extracts graph-edit labels that reconstruct the reactant graphs.

Molecular Transformer-style tokenized USPTO-MIT data is also available here:

- Molecular Transformer documentation: https://github.com/pschwllr/MolecularTransformer#pre-processing
- Tokenized data folder: https://ibm.ent.box.com/v/MolecularTransformerData

The tokenized Molecular Transformer data may be useful for forward reaction-prediction baselines, but it should not be assumed to be directly KGCL-ready. Standard tokenized forward-prediction files may be missing atom maps or may use layouts intended for sequence-to-sequence product prediction rather than graph-edit retrosynthesis. Before running KGCL canonicalization or preprocessing, verify that the reactions are atom-mapped. Do not commit the downloaded archive or extracted dataset to this repository.

Example local-only download commands:

```bash
mkdir -p data/downloads/uspto_mit

curl -L \
  -o data/downloads/uspto_mit/uspto_480k_atom_mapped.zip \
  https://github.com/wengong-jin/nips17-rexgen/raw/master/USPTO/data.zip

unzip -q \
  data/downloads/uspto_mit/uspto_480k_atom_mapped.zip \
  -d data/downloads/uspto_mit/atom_mapped
```

The current `kgcl-import-uspto-mit` command imports Molecular Transformer / Graph2SMILES-style `src-*.txt` plus `tgt-*.txt` files, or split CSV files named `train.csv`, `val.csv`/`valid.csv`, and `test.csv`; it does not directly consume the `nips17-rexgen` zip archive layout. After extracting the archive, convert its train/dev/test reaction files into KGCL raw CSV files before canonicalization/preprocessing:

- `data/uspto_mit/raw_train.csv`
- `data/uspto_mit/raw_val.csv`
- `data/uspto_mit/raw_test.csv`
- required reaction column: `reactants>reagents>production`
- required value format: `reactants>>product`

Keep `data/uspto_mit/` and `data/downloads/` as local working directories only.

## Running KGCL on USPTO-MIT / USPTO-480K

USPTO-MIT / USPTO-480K is commonly distributed for forward reaction prediction. This repository supports it as `uspto_mit` only in KGCL retrosynthesis-by-inversion mode: the model input is the product graph, the ground-truth label is the precursor/reactant side, and internal reaction strings are stored as `reactants>>product` in the existing `reactants>reagents>production` column. Results from this mode are not directly comparable to forward USPTO-480K product-prediction benchmarks.

Atom-mapped reactions are required for graph-edit label generation. Standard unmapped USPTO-MIT text data must be atom-mapped before canonicalization/preprocessing, or explicitly skipped with `--allow-unmapped-skip` during import. RXNMapper is not added as a required dependency. Reagents are dropped by default when separate reagent fields exist; pass `--include-reagents-as-reactants` only if your evaluation protocol intentionally treats reagents as precursors.

Supported inputs include Molecular Transformer / Graph2SMILES text layouts (`src-train.txt` / `tgt-train.txt`, `src-val.txt` / `tgt-val.txt`, `src-test.txt` / `tgt-test.txt`) and CSV files containing `rxn_smiles`, `reactants`, `reagents`, `product`, or `reactants>reagents>production` columns. The importer writes `data/uspto_mit/raw_train.csv`, `raw_val.csv`, and `raw_test.csv`.

```bash
python -m pip install -e ".[dev,chem]"

kgcl-import-uspto-mit \
  --input data/MIT_mapped \
  --output data/uspto_mit \
  --source-format auto

kgcl-canonicalize --dataset uspto_mit --mode train
kgcl-canonicalize --dataset uspto_mit --mode valid
kgcl-canonicalize --dataset uspto_mit --mode test

kgcl-preprocess --dataset uspto_mit --mode train
kgcl-preprocess --dataset uspto_mit --mode valid
kgcl-preprocess --dataset uspto_mit --mode test

kgcl-prepare-data --dataset uspto_mit --mode train --batch_size 256
kgcl-prepare-data --dataset uspto_mit --mode valid --batch_size 256
kgcl-prepare-data --dataset uspto_mit --mode test --batch_size 256

kgcl-train --dataset uspto_mit --lr 1e-4 --epochs 200 --num_workers 4

kgcl-eval --dataset uspto_mit --checkpoint auto
```
