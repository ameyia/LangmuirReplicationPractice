import csv

# Merge the manually-verified complexes with the auto-searched ones,
# dropping the extra molecule_a/molecule_b columns, and de-duplicating.

def load_rows(path, has_molecule_cols=False):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["pdb_id"], r["chain_a"], r["chain_b"]))
    return rows

manual = [("1BRS","A","D"), ("1BRS","B","E"), ("1BRS","C","F"), ("2PTC","E","I")]
auto = load_rows("complexes.csv")  # your auto-search output

combined = list(dict.fromkeys(manual + auto))  # dedupe, preserve order

with open("complexes_merged.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["pdb_id", "chain_a", "chain_b"])
    writer.writerows(combined)

print(f"Merged {len(combined)} unique complexes into complexes_merged.csv")