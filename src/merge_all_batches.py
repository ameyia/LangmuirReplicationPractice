import csv
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIST_DIR = PROJECT_ROOT / "data/complex_lists"

def load_rows(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["pdb_id"], r["chain_a"], r["chain_b"]))
    return rows

# Reads historical root lists and all keyword batches. The generated output is
# excluded so rerunning this script does not feed the output back into itself.
all_rows = []
input_paths = (
    glob.glob(str(LIST_DIR / "batch*.csv"))
    + glob.glob(str(LIST_DIR / "complexes*.csv"))
    + glob.glob(str(LIST_DIR / "batches/*.csv"))
    + [str(LIST_DIR / "manual_additions.csv")]
)
output_path = LIST_DIR / "complexes_full.csv"
for path in input_paths:
    if Path(path) == output_path:
        continue
    print(f"Reading {path}...")
    all_rows.extend(load_rows(path))

combined = list(dict.fromkeys(all_rows))  # dedupe, preserve order

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["pdb_id", "chain_a", "chain_b"])
    writer.writerows(combined)

print(f"\nMerged {len(all_rows)} rows down to {len(combined)} unique complexes -> {output_path}")
