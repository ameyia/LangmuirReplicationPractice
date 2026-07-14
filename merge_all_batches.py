import csv
import glob

def load_rows(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["pdb_id"], r["chain_a"], r["chain_b"]))
    return rows

# Reads every CSV matching batch*.csv or complexes*.csv in the current folder
all_rows = []
for path in glob.glob("batch*.csv") + glob.glob("complexes*.csv") + glob.glob("batches/*.csv"):
    print(f"Reading {path}...")
    all_rows.extend(load_rows(path))

combined = list(dict.fromkeys(all_rows))  # dedupe, preserve order

with open("complexes_full.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["pdb_id", "chain_a", "chain_b"])
    writer.writerows(combined)

print(f"\nMerged {len(all_rows)} rows down to {len(combined)} unique complexes -> complexes_full.csv")