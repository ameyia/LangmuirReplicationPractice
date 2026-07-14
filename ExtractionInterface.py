"""
extract_interface.py

Extracts interface residues between two chains of a PDB structure,
using the same logic as Song & Zhang 2024 (Langmuir) Section 2.2:
residues are considered part of the binding interface if any atom
pair between the two chains falls within a distance cutoff (paper
uses 6-7 Angstroms, citing closest-atom-pair distance).

Usage:
    python extract_interface.py 1BRS A D --cutoff 6.0

Requires:
    pip install biopython --break-system-packages
"""

import argparse
import sys
from Bio.PDB import PDBList, PDBParser
from Bio.PDB.Polypeptide import is_aa
import numpy as np

# 3-letter -> 1-letter amino acid code, plus hydroxyproline (HYP)
# which the paper explicitly includes as a 21st "residue type"
# because it appears in collagen.
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'HYP': 'O',  # hydroxyproline - non-standard, common in collagen
}


def download_structure(pdb_id, out_dir="."):
    """Download a PDB structure by ID using Biopython's PDBList."""
    pdbl = PDBList()
    # PDBList saves as pdb<id>.ent by default
    path = pdbl.retrieve_pdb_file(pdb_id, pdir=out_dir, file_format="pdb")
    return path


def get_residue_atoms(chain):
    """Return list of (residue, list_of_atom_coords) for standard
    amino acid residues in a chain, skipping waters/heteroatoms."""
    residues = []
    for res in chain:
        if is_aa(res, standard=False) and res.get_resname() in THREE_TO_ONE:
            coords = np.array([atom.get_coord() for atom in res])
            residues.append((res, coords))
    return residues


def find_interface_residues(chain_a, chain_b, cutoff=6.0):
    """
    For every residue pair (one from chain_a, one from chain_b),
    compute the minimum atom-atom distance. If it's <= cutoff,
    both residues are flagged as interface residues.

    Returns two lists: interface residues from chain_a, chain_b,
    each as (resnum, resname_1letter).
    """
    res_a = get_residue_atoms(chain_a)
    res_b = get_residue_atoms(chain_b)

    interface_a = set()
    interface_b = set()

    for res_a_obj, coords_a in res_a:
        for res_b_obj, coords_b in res_b:
            # pairwise distance matrix between all atoms in res_a and res_b
            diffs = coords_a[:, None, :] - coords_b[None, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=-1))
            min_dist = dists.min()

            if min_dist <= cutoff:
                resname_a = THREE_TO_ONE[res_a_obj.get_resname()]
                resname_b = THREE_TO_ONE[res_b_obj.get_resname()]
                interface_a.add((res_a_obj.get_id()[1], resname_a))
                interface_b.add((res_b_obj.get_id()[1], resname_b))

    return sorted(interface_a), sorted(interface_b)


def main():
    parser = argparse.ArgumentParser(description="Extract interface residues between two PDB chains.")
    parser.add_argument("pdb_id", help="4-character PDB ID, e.g. 1BRS")
    parser.add_argument("chain_a", help="First chain ID, e.g. A")
    parser.add_argument("chain_b", help="Second chain ID, e.g. D")
    parser.add_argument("--cutoff", type=float, default=6.0,
                         help="Distance cutoff in Angstroms (paper uses 6-7 A)")
    parser.add_argument("--pdb_file", default=None,
                         help="Optional: path to a local .pdb file instead of downloading")
    args = parser.parse_args()

    if args.pdb_file:
        pdb_path = args.pdb_file
    else:
        print(f"Downloading {args.pdb_id}...")
        pdb_path = download_structure(args.pdb_id)

    parser_bio = PDBParser(QUIET=True)
    structure = parser_bio.get_structure(args.pdb_id, pdb_path)
    model = structure[0]  # first model (crystal structures usually have just one)

    chain_a = model[args.chain_a]
    chain_b = model[args.chain_b]

    interface_a, interface_b = find_interface_residues(chain_a, chain_b, cutoff=args.cutoff)

    print(f"\n--- Interface residues (cutoff = {args.cutoff} A) ---")
    print(f"\nChain {args.chain_a}: {len(interface_a)} interface residues")
    for resnum, resname in interface_a:
        print(f"  {resname}{resnum}")

    print(f"\nChain {args.chain_b}: {len(interface_b)} interface residues")
    for resnum, resname in interface_b:
        print(f"  {resname}{resnum}")

    # Also print as plain sequences - this is the format the paper's
    # dataset actually needs: two amino acid sequences per data point.
    seq_a = "".join(resname for _, resname in interface_a)
    seq_b = "".join(resname for _, resname in interface_b)
    print(f"\nChain {args.chain_a} interface sequence: {seq_a}")
    print(f"Chain {args.chain_b} interface sequence: {seq_b}")


if __name__ == "__main__":
    main()