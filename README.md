# Langmuir replication practice

This repository is a handoff-ready record of an attempt to reproduce the
protein-interface sequence workflow and partner-prediction model described in
the IDProMat/Langmuir project.

Start with [docs/PROJECT_HANDOFF.md](docs/PROJECT_HANDOFF.md). It explains what
was tried, what each artifact means, the observed results, and the unresolved
limitations.

## Project layout

```text
data/
  complex_lists/       PDB/chain candidates and 167 keyword-search batches
  extracted/           Interface-sequence dataset and extraction failures
docs/                  Handoff notes and data dictionary
notebooks/             Colab launcher notebook (currently only a stub)
outputs/
  logs/                Four preserved training/run logs
  models/              Intended location for checkpoints (none are present)
src/                   Search, extraction, merge, and model-training code
structures/
  pdb_cache/            2,525 cached PDB structure files (~1.4 GB)
  1BRS.pdb              Early manually downloaded example
```

The whole repository is located in the UCB OneDrive-synced research folder.
Whether another person can see it still depends on the parent folder's
OneDrive sharing permissions.

## Quick reproduction

Create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Regenerate the combined list from the preserved search batches:

```bash
python src/merge_all_batches.py
```

Re-extract interface sequences (uses the existing PDB cache and may download
missing structures):

```bash
python src/batch_extract_interfaces.py \
  data/complex_lists/complexes_full.csv
```

Train the model:

```bash
python src/idpromat_model.py data/extracted/dataset.csv \
  --epochs 2000 --batch_size 32
```

The default checkpoint path is `outputs/models/best_model.pt`.

## Important cautions

- The candidate-generation script automatically chooses the first two distinct
  protein molecule names returned by RCSB. Those choices were not all manually
  validated.
- The 2,420-row extracted dataset and 144-row failure file account for all
  2,564 candidate triples.
- No trained `.pt` checkpoint is in this handoff. Only logs of past runs remain.
- The notebook is not a complete analysis; it contains a Colab badge and an
  empty code cell.

