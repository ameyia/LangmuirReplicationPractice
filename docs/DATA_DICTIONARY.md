# Data dictionary and provenance

## `data/complex_lists/`

| Artifact | Rows/files | Meaning |
|---|---:|---|
| `complexes_full.csv` | 2,564 rows | Deduplicated master list of PDB/chain triples |
| `batches/` | 167 CSV files | RCSB search results by keyword category |
| `complexes.csv` | 20 rows | Early search output with molecule names |
| `complexes_auto.csv` | 4 rows | Later small automatic-search output |
| `complexes_merged.csv` | 23 rows | Early manual + search merge |
| `batch2.csv`–`batch5.csv` | 82 rows total | Historical small search batches |
| `batch_collagen.csv` | 19 rows | Historical collagen-focused batch |
| `manual_additions.csv` | 6 rows | Triples found only in the historical master list |

Complex-list columns:

- `pdb_id`: four-character (or newer extended) PDB entry identifier.
- `chain_a`, `chain_b`: chain identifiers chosen as the interacting pair.
- `molecule_a`, `molecule_b`: descriptive RCSB molecule names when retained.

`complexes_full.csv` is generated from the other lists (including the explicit
manual additions) by
`src/merge_all_batches.py`; it drops molecule descriptions and deduplicates
exact PDB/chain triples.

## `data/extracted/`

`dataset.csv` contains 2,420 successful extractions:

- `pdb_id`, `chain_a`, `chain_b`: source complex and chain pair.
- `seq_a`, `seq_b`: interface residue sequences for the two chains.
- `n_residues_a`, `n_residues_b`: sequence lengths.

An interface residue is an amino-acid residue with at least one atom no more
than 6.0 Å from an atom in the partner chain. Sequences are sorted by residue
number; they are interface-residue strings, not necessarily continuous
segments of the full protein sequence.

`dataset_failures.csv` contains 144 failed triples:

- 106: no interface found within 6 Å.
- 38: other errors, retained verbatim in the `error` column.

Together the successful and failed files account for all 2,564 master-list
triples.

## `structures/`

- `pdb_cache/`: 2,525 structure files downloaded by Biopython (~1.4 GB).
- `1BRS.pdb`: separate early example structure.

The cache is reproducible from PDB in principle, but retaining it makes the
handoff robust to network changes and avoids repeated downloads.

## `outputs/`

`logs/` preserves four historical logs. `models/` is intentionally ready for
future checkpoints, but no model checkpoint was found during reorganization.
