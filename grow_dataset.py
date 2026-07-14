"""
grow_dataset.py

Runs search_and_populate_complexes.py across a whole list of keywords
automatically, so you don't have to type each command by hand.
Saves one CSV per keyword into the batches/ folder.

Usage:
    python3 grow_dataset.py

Edit the KEYWORDS list below to change what it searches for.
Requires search_and_populate_complexes.py in the same folder.
"""

import subprocess
import os

KEYWORDS = [
    # Antibody and immune complexes
    "antibody antigen complex",
    "antibody peptide complex",
    "nanobody antigen complex",
    "T cell receptor peptide MHC",
    "immune receptor ligand",

    # Enzyme and inhibitor complexes
    "protease peptide inhibitor",
    "serine protease inhibitor",
    "thrombin peptide inhibitor",
    "factor Xa inhibitor peptide",
    "kinase regulatory protein complex",
    "enzyme inhibitor protein complex",

    # Receptor and ligand complexes
    "receptor ligand protein complex",
    "cytokine receptor complex",
    "growth factor receptor complex",
    "integrin ligand complex",
    "integrin collagen complex",
    "cell adhesion receptor ligand",
    "GPCR peptide ligand complex",

    # Coagulation and platelet-related
    "coagulation factor complex",
    "platelet receptor ligand",
    "thrombin hirudin complex",
    "fibrinogen receptor complex",
    "von Willebrand factor receptor",
    "collagen binding protein complex",
    "GPVI collagen complex",
    "antithrombin thrombin complex",

    # General protein–protein complexes
    "protein protein complex",
    "heterodimer protein complex",
    "enzyme substrate complex",
    "regulatory protein complex",
    "transcription factor cofactor complex",
    "signaling protein complex",
    "scaffold protein complex",
    "protein peptide complex",

    # Host–pathogen interactions
    "viral protein antibody complex",
    "viral receptor complex",
    "bacterial toxin antitoxin",
    "toxin receptor complex",
    "host pathogen protein complex",

    # Classic interface categories
    "barnase barstar",
    "ubiquitin binding protein",
    "SH2 domain peptide complex",
    "SH3 domain peptide complex",
    "PDZ domain peptide complex",
    "calmodulin peptide complex",

    None,  # general variety
]
LIMIT = 25
MAX_RESOLUTION = 2.5

os.makedirs("batches", exist_ok=True)

for kw in KEYWORDS:
    safe_name = kw.replace(" ", "_") if kw else "general"
    out_path = f"batches/batch_{safe_name}.csv"

    cmd = [
        "python3", "search_and_populate_complexes.py",
        "--max_resolution", str(MAX_RESOLUTION),
        "--limit", str(LIMIT),
        "--out", out_path,
    ]
    if kw:
        cmd += ["--keyword", kw]

    print(f"\n=== Running: {kw or '(no keyword)'} ===")
    subprocess.run(cmd)

print("\nAll searches done. Run merge_all_batches.py next (point it at the batches/ folder).")
