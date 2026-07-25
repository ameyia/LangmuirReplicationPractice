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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
BATCH_DIR = PROJECT_ROOT / "data/complex_lists/batches"

KEYWORDS = [
    # --------------------------------------------------
    # Antibody and immune-recognition complexes
    # --------------------------------------------------
    "antibody antigen complex",
    "antibody peptide complex",
    "antibody protein complex",
    "nanobody antigen complex",
    "single domain antibody antigen",
    "Fab antigen complex",
    "immune receptor ligand complex",
    "T cell receptor peptide MHC",
    "T cell receptor MHC complex",
    "peptide MHC complex",
    "MHC antibody complex",
    "B cell receptor antigen",
    "complement protein complex",
    "complement antibody complex",
    "Fc receptor antibody complex",

    # --------------------------------------------------
    # Enzyme–inhibitor and enzyme–substrate complexes
    # --------------------------------------------------
    "protease peptide inhibitor",
    "protease protein inhibitor",
    "serine protease inhibitor complex",
    "cysteine protease inhibitor complex",
    "metalloprotease inhibitor complex",
    "aspartic protease inhibitor complex",
    "thrombin peptide inhibitor",
    "factor Xa peptide inhibitor",
    "factor IXa inhibitor complex",
    "plasmin inhibitor complex",
    "trypsin peptide inhibitor",
    "enzyme substrate protein complex",
    "enzyme regulatory protein complex",
    "kinase peptide substrate complex",
    "kinase peptide inhibitor complex",
    "kinase regulatory protein complex",
    "phosphatase regulatory protein complex",
    "ubiquitin ligase substrate complex",
    "deubiquitinase ubiquitin complex",

    # --------------------------------------------------
    # Receptor–ligand complexes
    # --------------------------------------------------
    "receptor ligand protein complex",
    "receptor peptide ligand complex",
    "cytokine receptor complex",
    "chemokine receptor complex",
    "growth factor receptor complex",
    "hormone receptor peptide complex",
    "integrin ligand complex",
    "integrin collagen complex",
    "integrin fibronectin complex",
    "integrin fibrinogen complex",
    "cell adhesion receptor ligand",
    "GPCR peptide ligand complex",
    "receptor tyrosine kinase ligand complex",
    "death receptor ligand complex",
    "Notch ligand complex",
    "semaphorin receptor complex",
    "ephrin receptor complex",

    # --------------------------------------------------
    # Coagulation, thrombosis, and platelet complexes
    # --------------------------------------------------
    "coagulation factor complex",
    "coagulation factor peptide inhibitor",
    "platelet receptor ligand",
    "platelet glycoprotein ligand complex",
    "thrombin hirudin complex",
    "thrombin thrombomodulin complex",
    "thrombin antithrombin complex",
    "factor Xa antithrombin complex",
    "factor VIII factor IX complex",
    "factor VII tissue factor complex",
    "fibrinogen receptor complex",
    "fibrin peptide complex",
    "von Willebrand factor receptor complex",
    "von Willebrand factor platelet complex",
    "collagen binding protein complex",
    "collagen peptide protein complex",
    "GPVI collagen complex",
    "GPIb von Willebrand factor",
    "plasminogen activator inhibitor complex",
    "protein C thrombomodulin complex",
    "tissue factor pathway inhibitor",
    "alpha2 beta1 collagen complex",

    # --------------------------------------------------
    # Protein–peptide recognition domains
    # --------------------------------------------------
    "protein peptide complex",
    "peptide binding domain complex",
    "PDZ domain peptide complex",
    "SH2 domain peptide complex",
    "SH3 domain peptide complex",
    "WW domain peptide complex",
    "PH domain peptide complex",
    "14-3-3 peptide complex",
    "calmodulin peptide complex",
    "cyclin kinase peptide complex",
    "bromodomain peptide complex",
    "chromodomain peptide complex",
    "FHA domain peptide complex",
    "Polo box domain peptide",
    "LIM domain peptide complex",
    "armadillo repeat peptide complex",
    "ankyrin repeat peptide complex",
    "leucine rich repeat ligand complex",
    "TPR repeat peptide complex",

    # --------------------------------------------------
    # Signaling and regulatory complexes
    # --------------------------------------------------
    "signaling protein complex",
    "regulatory protein complex",
    "scaffold protein complex",
    "adaptor protein complex",
    "G protein receptor complex",
    "small GTPase effector complex",
    "Ras effector complex",
    "Rho GTPase effector complex",
    "Rab effector complex",
    "transcription factor cofactor complex",
    "transcriptional repressor complex",
    "transcriptional activator complex",
    "nuclear receptor coactivator peptide",
    "apoptosis protein complex",
    "Bcl-2 BH3 peptide complex",
    "caspase inhibitor complex",

    # --------------------------------------------------
    # Transport and membrane-associated complexes
    # --------------------------------------------------
    "membrane protein peptide complex",
    "ion channel toxin complex",
    "ion channel regulatory protein complex",
    "transporter accessory protein complex",
    "porin binding protein complex",
    "SNARE complex",
    "vesicle fusion protein complex",
    "membrane receptor ectodomain ligand",
    "lipoprotein receptor ligand complex",

    # --------------------------------------------------
    # Host–pathogen and toxin complexes
    # --------------------------------------------------
    "viral protein antibody complex",
    "viral peptide antibody complex",
    "viral receptor complex",
    "viral glycoprotein receptor complex",
    "virus host protein complex",
    "bacterial toxin antitoxin",
    "toxin receptor complex",
    "toxin antibody complex",
    "host pathogen protein complex",
    "bacterial effector host protein",
    "parasite host protein complex",
    "fungal protein host receptor",

    # --------------------------------------------------
    # Protein-processing and degradation complexes
    # --------------------------------------------------
    "ubiquitin binding protein",
    "ubiquitin receptor complex",
    "SUMO protein complex",
    "proteasome regulator complex",
    "chaperone client protein complex",
    "heat shock protein cochaperone",
    "ribosomal protein complex",
    "translation factor ribosome complex",

    # --------------------------------------------------
    # Extracellular matrix and adhesion
    # --------------------------------------------------
    "extracellular matrix protein complex",
    "collagen receptor complex",
    "fibronectin receptor complex",
    "laminin receptor complex",
    "elastin binding protein complex",
    "cadherin complex",
    "selectin ligand complex",
    "cell adhesion molecule complex",
    "syndecan ligand complex",
    "heparin binding protein complex",
    "heparan sulfate protein complex",

    # --------------------------------------------------
    # Classic benchmark interactions
    # --------------------------------------------------
    "barnase barstar",
    "trypsin bovine pancreatic trypsin inhibitor",
    "ribonuclease inhibitor complex",
    "lysozyme antibody complex",
    "calmodulin target peptide",
    "streptavidin peptide complex",

    # --------------------------------------------------
    # Broad searches for variety
    # --------------------------------------------------
    "protein protein complex",
    "heterodimer protein complex",
    "heterooligomer protein complex",
    "binary protein complex",
    "protein interaction complex",
    "biological assembly protein complex",
    None,
]
LIMIT = 25
MAX_RESOLUTION = 2.5

BATCH_DIR.mkdir(parents=True, exist_ok=True)

for kw in KEYWORDS:
    safe_name = kw.replace(" ", "_") if kw else "general"
    out_path = BATCH_DIR / f"batch_{safe_name}.csv"

    cmd = [
        "python3", str(SCRIPT_DIR / "search_and_populate_complexes.py"),
        "--max_resolution", str(MAX_RESOLUTION),
        "--limit", str(LIMIT),
        "--out", str(out_path),
    ]
    if kw:
        cmd += ["--keyword", kw]

    print(f"\n=== Running: {kw or '(no keyword)'} ===")
    subprocess.run(cmd)

print("\nAll searches done. Run merge_all_batches.py next (point it at the batches/ folder).")
