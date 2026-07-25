"""
batch_extract_interfaces.py

Batch version of extract_interface.py. Takes a list of
(pdb_id, chain_a, chain_b) triples, runs interface extraction on
each one, and writes a single structured dataset file - the same
format described in the paper's Section 2.2: one row per protein
complex, containing the two interacting amino acid sequences.

Failures (missing chain, download error, no interface found, etc.)
are logged and skipped rather than crashing the whole run, since
with a real batch of PDB structures you WILL hit messy cases.

Usage:
    python src/batch_extract_interfaces.py data/complex_lists/complexes_full.csv

Where complexes.csv has columns: pdb_id,chain_a,chain_b

Requires:
    pip install biopython --break-system-packages
"""

import argparse
import csv
import os
import sys
from pathlib import Path
import numpy as np
from Bio.PDB import PDBList, PDBParser
from Bio.PDB.Polypeptide import is_aa

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'HYP': 'O',
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download_structure(pdb_id, out_dir="pdb_cache"):
    """Download a PDB structure, caching it locally so re-runs are fast."""
    os.makedirs(out_dir, exist_ok=True)
    pdbl = PDBList()
    path = pdbl.retrieve_pdb_file(pdb_id, pdir=out_dir, file_format="pdb")
    return path


def get_residue_atoms(chain):
    residues = []
    for res in chain:
        if is_aa(res, standard=False) and res.get_resname() in THREE_TO_ONE:
            coords = np.array([atom.get_coord() for atom in res])
            residues.append((res, coords))
    return residues


def find_interface_residues(chain_a, chain_b, cutoff=6.0):
    res_a = get_residue_atoms(chain_a)
    res_b = get_residue_atoms(chain_b)

    interface_a = set()
    interface_b = set()

    for res_a_obj, coords_a in res_a:
        for res_b_obj, coords_b in res_b:
            diffs = coords_a[:, None, :] - coords_b[None, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=-1))
            if dists.min() <= cutoff:
                interface_a.add((res_a_obj.get_id()[1], THREE_TO_ONE[res_a_obj.get_resname()]))
                interface_b.add((res_b_obj.get_id()[1], THREE_TO_ONE[res_b_obj.get_resname()]))

    return sorted(interface_a), sorted(interface_b)


def process_one_complex(pdb_id, chain_a_id, chain_b_id, cutoff, cache_dir, parser_bio):
    """Returns (seq_a, seq_b, n_res_a, n_res_b) on success, or raises on failure."""
    pdb_path = download_structure(pdb_id, out_dir=cache_dir)
    structure = parser_bio.get_structure(pdb_id, pdb_path)
    model = structure[0]

    if chain_a_id not in model:
        raise ValueError(f"Chain {chain_a_id} not found in {pdb_id} (available: {[c.id for c in model]})")
    if chain_b_id not in model:
        raise ValueError(f"Chain {chain_b_id} not found in {pdb_id} (available: {[c.id for c in model]})")

    chain_a = model[chain_a_id]
    chain_b = model[chain_b_id]

    interface_a, interface_b = find_interface_residues(chain_a, chain_b, cutoff=cutoff)

    if len(interface_a) == 0 or len(interface_b) == 0:
        raise ValueError(f"No interface found within {cutoff} A cutoff")

    seq_a = "".join(r for _, r in interface_a)
    seq_b = "".join(r for _, r in interface_b)
    return seq_a, seq_b, len(interface_a), len(interface_b)


def main():
    parser = argparse.ArgumentParser(description="Batch-extract interface residue sequences from a list of PDB complexes.")
    parser.add_argument("complexes_csv", help="CSV file with columns: pdb_id,chain_a,chain_b")
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data/extracted/dataset.csv"))
    parser.add_argument("--cache_dir", default=str(PROJECT_ROOT / "structures/pdb_cache"))
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.complexes_csv) as f:
        reader = csv.DictReader(f)
        jobs = list(reader)

    parser_bio = PDBParser(QUIET=True)

    results = []
    failures = []

    for i, job in enumerate(jobs, 1):
        pdb_id = job["pdb_id"].strip()
        chain_a = job["chain_a"].strip()
        chain_b = job["chain_b"].strip()

        print(f"[{i}/{len(jobs)}] {pdb_id} ({chain_a}, {chain_b})...", end=" ")
        try:
            seq_a, seq_b, n_a, n_b = process_one_complex(
                pdb_id, chain_a, chain_b, args.cutoff, args.cache_dir, parser_bio
            )
            results.append({
                "pdb_id": pdb_id, "chain_a": chain_a, "chain_b": chain_b,
                "seq_a": seq_a, "seq_b": seq_b,
                "n_residues_a": n_a, "n_residues_b": n_b,
            })
            print(f"OK ({n_a} + {n_b} interface residues)")
        except Exception as e:
            failures.append({"pdb_id": pdb_id, "chain_a": chain_a, "chain_b": chain_b, "error": str(e)})
            print(f"FAILED: {e}")

    # Write successful results
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pdb_id", "chain_a", "chain_b", "seq_a", "seq_b", "n_residues_a", "n_residues_b"])
        writer.writeheader()
        writer.writerows(results)

    # Write failure log separately so nothing is silently lost
    fail_path = args.out.replace(".csv", "_failures.csv")
    if failures:
        with open(fail_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["pdb_id", "chain_a", "chain_b", "error"])
            writer.writeheader()
            writer.writerows(failures)

    print(f"\nDone. {len(results)} succeeded, {len(failures)} failed.")
    print(f"Dataset written to: {args.out}")
    if failures:
        print(f"Failure log written to: {fail_path}")


if __name__ == "__main__":
    main()
