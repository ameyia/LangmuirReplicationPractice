"""
dock_peptides.py

Dock short candidate peptides onto a collagen triple helix with AutoDock Vina,
and compare each one against length-matched SCRAMBLED controls.

Why the controls: Vina scores depend heavily on peptide length and composition
(longer / more aromatic / more charged peptides score "better" for boring reasons).
So a raw score means little. What matters is how much better a candidate does
than scrambled versions of ITSELF (same residues, shuffled order).

Default receptor: PDB 1DZI (integrin alpha2 I domain bound to a GFOGER collagen
peptide). We keep only the collagen-like chains (auto-detected by their
Gly/Pro/Hyp-rich sequence), drop the integrin, and place the search box around
the collagen residues the integrin contacted (a real platelet/cell-adhesion site).

LIMITS (read these):
  * Vina handles flexible peptides badly beyond ~6-8 residues. Keep candidates short.
  * Hydroxyproline (O) is not buildable here; O is replaced by P in the ligand.
  * A docking score is NOT a measure of antithrombotic activity. It only asks
    "does this sit well on this patch of collagen?".

Setup (Mac, miniforge base env):
    conda install -c conda-forge vina openbabel rdkit biopython
    pip install meeko gemmi
  (check that `vina --version` works; if not, put a Vina 1.2.5 binary on your PATH
   or pass --vina_path /path/to/vina)

Usage (from src/):
    python dock_peptides.py --peptides GFPGER GPPGPP DDEEDD
    python dock_peptides.py --peptide_file my_candidates.txt --exhaustiveness 16
"""

import argparse
import csv
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from Bio.PDB import NeighborSearch, PDBIO, PDBList, PDBParser, Select
from rdkit import Chem
from rdkit.Chem import AllChem

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "HYP": "O",
}
STANDARD20 = set("ACDEFGHIKLMNPQRSTVWY")


# ----------------------------------------------------------------------------
# Receptor
# ----------------------------------------------------------------------------
def chain_sequence(chain):
    return "".join(THREE_TO_ONE[r.get_resname()] for r in chain if r.get_resname() in THREE_TO_ONE)


def is_collagen_like(seq):
    """Collagen strands are ~1/3 Gly with lots of Pro/Hyp (O)."""
    if len(seq) < 12:
        return False
    f_gpo = sum(c in "GPO" for c in seq) / len(seq)
    f_g = seq.count("G") / len(seq)
    return f_gpo >= 0.6 and f_g >= 0.25


class KeepChains(Select):
    def __init__(self, chain_ids):
        self.chain_ids = set(chain_ids)

    def accept_chain(self, chain):
        return chain.id in self.chain_ids

    def accept_residue(self, residue):
        return residue.get_resname() in THREE_TO_ONE  # drops waters and ligands, keeps HYP

    def accept_atom(self, atom):
        return (not atom.is_disordered()) or atom.get_altloc() in ("A", " ")


def prepare_receptor(args, workdir):
    """Returns (receptor_pdbqt, box_center, box_size, description)."""
    parser = PDBParser(QUIET=True)
    if args.receptor_pdb:
        pdb_path = args.receptor_pdb
    else:
        pdb_path = PDBList(verbose=False).retrieve_pdb_file(
            args.pdb_id, pdir=str(workdir), file_format="pdb")
    structure = parser.get_structure("rec", pdb_path)
    model = structure[0]

    chain_seqs = {c.id: chain_sequence(c) for c in model}
    if args.receptor_chains:
        rec_ids = list(args.receptor_chains)
    else:
        rec_ids = [cid for cid, s in chain_seqs.items() if is_collagen_like(s)]
    if not rec_ids:
        sys.exit(f"Could not find collagen-like chains automatically. Chains seen: "
                 f"{ {k: v[:25] for k, v in chain_seqs.items()} }. Use --receptor_chains.")
    partner_ids = [cid for cid in chain_seqs if cid not in rec_ids and chain_seqs[cid]]
    if args.partner_chains:
        partner_ids = list(args.partner_chains)
    print(f"Receptor chains: {rec_ids}   (sequence of first: {chain_seqs[rec_ids[0]][:30]}...)")
    print(f"Partner chains used only to locate the binding site: {partner_ids}")

    # --- locate box ---
    rec_atoms = [a for cid in rec_ids for a in model[cid].get_atoms()]
    if args.box_center and args.box_size:
        center, size = np.array(args.box_center, float), np.array(args.box_size, float)
        desc = "manual box"
    else:
        site_atoms = []
        if partner_ids and not args.blind:
            partner_atoms = [a for cid in partner_ids for a in model[cid].get_atoms()]
            ns = NeighborSearch(partner_atoms)
            for res in (r for cid in rec_ids for r in model[cid] if r.get_resname() in THREE_TO_ONE):
                if any(ns.search(a.coord, args.site_cutoff) for a in res):
                    site_atoms.extend(res.get_atoms())
        if site_atoms:
            coords = np.array([a.coord for a in site_atoms])
            desc = f"site contacted by partner chain(s) ({len(set(a.get_parent().id for a in site_atoms))} residues)"
        else:
            coords = np.array([a.coord for a in rec_atoms])
            desc = "whole receptor (blind docking)"
        center = (coords.max(0) + coords.min(0)) / 2
        size = np.clip(coords.max(0) - coords.min(0) + 2 * args.box_pad, args.min_box, args.max_box)
    print(f"Search box: {desc}\n  center = {np.round(center, 1)}  size = {np.round(size, 1)} A")

    # --- write receptor PDB (HYP made ATOM-like so openbabel keeps it) ---
    for cid in rec_ids:
        for res in model[cid]:
            if res.id[0] != " ":
                res.id = (" ", res.id[1], res.id[2])
    rec_pdb = Path(workdir) / "receptor.pdb"
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(rec_pdb), KeepChains(rec_ids))

    rec_pdbqt = Path(workdir) / "receptor.pdbqt"
    from openbabel import pybel
    mol = next(pybel.readfile("pdb", str(rec_pdb)))
    mol.addh()
    mol.write("pdbqt", str(rec_pdbqt), opt={"r": None}, overwrite=True)  # r = rigid receptor
    return rec_pdbqt, center, size, desc


# ----------------------------------------------------------------------------
# Ligand
# ----------------------------------------------------------------------------
def build_ligand_pdbqt(seq, out_path, seed=0):
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    mol = Chem.AddHs(Chem.MolFromSequence(seq))
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    setups = MoleculePreparation().prepare(mol)
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise RuntimeError(f"meeko failed for {seq}: {err}")
    Path(out_path).write_text(pdbqt)
    tors = next((int(l.split()[1]) for l in pdbqt.splitlines() if l.startswith("TORSDOF")), -1)
    return tors


# ----------------------------------------------------------------------------
# Vina
# ----------------------------------------------------------------------------
def run_vina(vina, rec, lig, center, size, out_pdbqt, exhaustiveness, cpu, seed):
    cmd = [vina, "--receptor", str(rec), "--ligand", str(lig), "--out", str(out_pdbqt),
           "--center_x", f"{center[0]:.3f}", "--center_y", f"{center[1]:.3f}", "--center_z", f"{center[2]:.3f}",
           "--size_x", f"{size[0]:.1f}", "--size_y", f"{size[1]:.1f}", "--size_z", f"{size[2]:.1f}",
           "--exhaustiveness", str(exhaustiveness), "--num_modes", "5", "--seed", str(seed)]
    if cpu:
        cmd += ["--cpu", str(cpu)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"vina failed:\n{r.stdout[-600:]}\n{r.stderr[-600:]}")
    for line in Path(out_pdbqt).read_text().splitlines():
        if line.startswith("REMARK VINA RESULT"):
            return float(line.split()[3])
    raise RuntimeError("no Vina result found in output")


def clean_sequence(s):
    s = s.strip().upper().replace("O", "P")
    bad = set(s) - STANDARD20
    if bad or not s:
        raise ValueError(f"unsupported residues {bad or '(empty)'} in '{s}'")
    return s


def dock_one(seq, ctx, tag):
    lig = ctx["work"] / f"lig_{tag}.pdbqt"
    out = ctx["work"] / f"out_{tag}.pdbqt"
    tors = build_ligand_pdbqt(seq, lig)
    score = run_vina(ctx["vina"], ctx["rec"], lig, ctx["center"], ctx["size"], out,
                     ctx["exh"], ctx["cpu"], ctx["seed"])
    return score, tors, out


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peptides", nargs="*", default=[], help="candidate sequences (O is treated as P)")
    ap.add_argument("--peptide_file", help="text file, one sequence per line (optional 'name seq')")
    ap.add_argument("--pdb_id", default="1DZI")
    ap.add_argument("--receptor_pdb", help="use a local PDB file instead of downloading")
    ap.add_argument("--receptor_chains", help="e.g. BCD; default: auto-detect collagen-like chains")
    ap.add_argument("--partner_chains", help="chains used only to locate the site; default: all others")
    ap.add_argument("--blind", action="store_true", help="box around the whole receptor instead of the partner site")
    ap.add_argument("--box_center", nargs=3, type=float)
    ap.add_argument("--box_size", nargs=3, type=float)
    ap.add_argument("--box_pad", type=float, default=8.0)
    ap.add_argument("--min_box", type=float, default=22.0)
    ap.add_argument("--max_box", type=float, default=36.0,
                    help="cap on box edge length (A); bigger boxes need much higher exhaustiveness")
    ap.add_argument("--site_cutoff", type=float, default=4.5)
    ap.add_argument("--n_controls", type=int, default=4, help="scrambled controls per candidate")
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--cpu", type=int, default=0, help="0 = all cores")
    ap.add_argument("--vina_path", default="vina")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_csv", default="docking_results.csv")
    ap.add_argument("--out_dir", default="docking_poses")
    ap.add_argument("--max_len_warn", type=int, default=8)
    args = ap.parse_args()

    if shutil.which(args.vina_path) is None and not Path(args.vina_path).exists():
        sys.exit("Vina not found. Install it (conda install -c conda-forge vina) or pass --vina_path.")

    cands = []
    for i, s in enumerate(args.peptides):
        cands.append((s.upper(), s))
    if args.peptide_file:
        for line in Path(args.peptide_file).read_text().splitlines():
            parts = line.split()
            if parts:
                cands.append((parts[0], parts[-1]) if len(parts) > 1 else (parts[0], parts[0]))
    if not cands:
        sys.exit("Give --peptides or --peptide_file.")

    work = Path(tempfile.mkdtemp(prefix="dock_"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working folder: {work}\n")
    rec, center, size, _ = prepare_receptor(args, work)
    ctx = dict(work=work, rec=rec, center=center, size=size, vina=args.vina_path,
               exh=args.exhaustiveness, cpu=args.cpu, seed=args.seed)

    rng = random.Random(args.seed)
    rows = []
    for k, (name, raw) in enumerate(cands, 1):
        try:
            seq = clean_sequence(raw)
        except ValueError as e:
            print(f"[{k}/{len(cands)}] skipping {name}: {e}")
            continue
        if len(seq) > args.max_len_warn:
            print(f"  WARNING: {seq} is {len(seq)} residues; Vina is unreliable for peptides this flexible.")
        print(f"[{k}/{len(cands)}] {name} ({seq}, {len(seq)} aa) ...", flush=True)
        score, tors, pose = dock_one(seq, ctx, f"c{k}")
        shutil.copy(pose, out_dir / f"{name}_best.pdbqt")

        ctrl_scores = []
        seen = {seq}
        tries = 0
        while len(ctrl_scores) < args.n_controls and tries < 50:
            tries += 1
            s = list(seq)
            rng.shuffle(s)
            s = "".join(s)
            if s in seen and len(set(seq)) > 1 and tries < 40:
                continue
            seen.add(s)
            sc, _, _ = dock_one(s, ctx, f"c{k}_ctl{len(ctrl_scores)}")
            ctrl_scores.append(sc)
        c_mean = statistics.mean(ctrl_scores) if ctrl_scores else float("nan")
        c_sd = statistics.pstdev(ctrl_scores) if len(ctrl_scores) > 1 else float("nan")
        delta = score - c_mean  # negative = better than its own scrambles
        rows.append(dict(name=name, sequence=seq, length=len(seq), torsions=tors,
                         score=score, ctrl_mean=round(c_mean, 2), ctrl_sd=round(c_sd, 2),
                         delta_vs_scrambled=round(delta, 2)))
        print(f"    score {score:.2f} | scrambled controls {c_mean:.2f} +/- {c_sd:.2f} | delta {delta:+.2f}")

        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\n" + "=" * 78)
    print("RESULTS (kcal/mol; more negative = better). Ranked by improvement over scrambled self.")
    print("=" * 78)
    print(f"{'name':<18}{'seq':<14}{'len':>4}{'score':>8}{'ctrl':>8}{'sd':>6}{'delta':>8}")
    for r in sorted(rows, key=lambda r: r["delta_vs_scrambled"]):
        print(f"{r['name'][:17]:<18}{r['sequence'][:13]:<14}{r['length']:>4}{r['score']:>8.2f}"
              f"{r['ctrl_mean']:>8.2f}{r['ctrl_sd']:>6.2f}{r['delta_vs_scrambled']:>+8.2f}")
    print("\nHow to read: a candidate only looks interesting if delta is clearly negative, roughly more than")
    print("2x the control sd AND at least ~0.5-1 kcal/mol. Vina's error is about 2 kcal/mol, so small")
    print("differences are noise. Poses are in", out_dir, "; results in", args.out_csv)


if __name__ == "__main__":
    main()