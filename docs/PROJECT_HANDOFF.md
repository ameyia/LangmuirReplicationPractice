# Project handoff: what was tried

## Goal

The project attempts to build paired amino-acid sequences from interacting
protein interfaces in PDB structures, then train a neural model to predict the
interface sequence of one binding partner from the other.

## Work completed

1. **Small/manual candidate lists were assembled.** Early CSV files contain
   selected PDB IDs, chain pairs, and molecule names. A barnase–barstar example
   (`1BRS`) was also downloaded manually.
2. **RCSB searching was automated.** `search_and_populate_complexes.py` queries
   RCSB, keeps protein entities, and selects representative chains for two
   distinct molecule names.
3. **The search was expanded across 167 keyword categories.**
   `grow_dataset.py` covers antibodies, enzyme–inhibitor pairs,
   receptor–ligand pairs, coagulation, recognition domains, signaling,
   host–pathogen systems, and broad protein-complex searches.
4. **Candidate lists were merged and deduplicated.** The current combined list
   contains 2,564 unique `(pdb_id, chain_a, chain_b)` triples. Six triples that
   existed only in the historical master output are now explicitly preserved
   in `manual_additions.csv`, so the master can be reproduced without reading
   its own prior output.
5. **Interface residues were extracted.** For each chain pair, residues with
   any atom within 6 Å of the other chain were collected in residue-number
   order. Twenty standard amino acids plus hydroxyproline (`O`) are supported.
6. **A paired dataset was produced.** Extraction succeeded for 2,420 unique
   chain triples from 2,418 unique PDB entries. All sequence fields are
   non-empty. The other 144 triples are preserved in the failure log.
7. **A PyTorch model was trained in several runs.** The code uses an
   encoder/decoder reconstruction objective plus an MLP that maps one encoded
   sequence to its partner representation. It uses a seeded 90/10 train and
   validation split.

## Results visible in the logs

The most informative run is
`outputs/logs/CoLabTraining1BeforeTraining_log3data.txt`:

- 2,420 pairs loaded; 2,178 train and 242 held out.
- CUDA training reached epoch 1,600 and stopped after five validation checks
  without improvement.
- Best logged held-out partner-prediction accuracy: **13.25%**.
- The log says the best checkpoint was reloaded before final metrics, but that
  checkpoint file is not present in this repository.

Two CPU runs also completed:

- `training_log.txt`: 300 epochs; final held-out partner prediction **9.25%**.
- `training_log2.txt`: 2,000 epochs; final held-out partner prediction
  **12.00%**; reconstruction accuracy was 50.42% train and 31.22% validation.

`training_log3.txt` is not a training run: it records two failed starts caused
by `ModuleNotFoundError: No module named 'torch'`.

## What did not work or remains uncertain

- Candidate chain selection is heuristic. The script warns that molecule
  columns must be reviewed, but there is no record showing that all 2,564
  chain pairs were manually verified as biologically meaningful interfaces.
- Of the 144 extraction failures, 106 report no interface within 6 Å. The
  remaining 38 include missing chains, parsing/download problems, and other
  structure-specific issues; they have not been systematically repaired.
- The best model checkpoint is missing. The Colab log points to
  `/content/drive/MyDrive/best_model.pt`, so it may have remained in the
  researcher's personal Google Drive.
- The Colab notebook is only a launcher stub and does not capture the actual
  training setup.
- There is no exact environment lock file, Git commit recorded in the logs, or
  documented source-paper version. `requirements.txt` records the direct
  Python dependencies but not the historical package versions.
- “Correct length used” in partner-prediction evaluation gives the decoder the
  target sequence length. This should be disclosed when comparing the result
  with other prediction systems.

## Recommended next steps

1. Locate and add the best Colab checkpoint, if it still exists.
2. Manually validate a stratified sample of candidate chain pairs and quantify
   false selections from the automated RCSB heuristic.
3. Categorize and retry the 38 non-distance extraction failures.
4. Check for leakage at the protein-family or sequence-similarity level. The
   current random pair split can place closely related complexes in both train
   and validation sets.
5. Add baseline metrics and a target-length prediction strategy before treating
   partner-prediction accuracy as a deployable result.
6. Record exact package versions and the paper/version being replicated.

## OneDrive handoff status

All project artifacts are now organized under this OneDrive-synced project
folder. This confirms location, not access: the owner should verify that the
intended collaborators have permission on the shared parent folder and that
OneDrive shows sync completion, especially for the ~1.4 GB PDB cache.
