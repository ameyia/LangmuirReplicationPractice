import csv, random, sys

def load_dataset(csv_path):
    pairs = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["pdb_id"], row["chain_a"], row["chain_b"], row["seq_a"], row["seq_b"]))
    return pairs

def split_train_val(pairs, val_fraction=0.1, seed=42):
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]

pairs = load_dataset(sys.argv[1])
# match the actual (seq_a, seq_b) tuple format train() uses internally
pair_tuples = [(p[3], p[4]) for p in pairs]
train_pairs, val_pairs = split_train_val(pair_tuples, val_fraction=0.1, seed=42)
val_set = set(val_pairs)

if len(sys.argv) > 2 and sys.argv[2] == "--list-val":
    print(f"{len(val_pairs)} pairs in the held-out validation set. Showing up to 15:\n")
    print(f"{'pdb_id':<10}{'chain_a':<9}{'chain_b':<9}{'len(a)':<8}{'len(b)':<8}seq_b (try this as --target)")
    shown = 0
    for p in pairs:
        tup = (p[3], p[4])
        if tup in val_set:
            print(f"{p[0]:<10}{p[1]:<9}{p[2]:<9}{len(p[3]):<8}{len(p[4]):<8}{p[4]}")
            shown += 1
            if shown >= 15:
                break
else:
    for p in pairs:
        if p[0].upper().endswith("1DZI") or p[0].upper() == "DZI":
            tup = (p[3], p[4])
            in_train = tup in train_pairs
            in_val = tup in val_pairs
            print(f"Found: {p}")
            print(f"In TRAIN split: {in_train}")
            print(f"In VAL split: {in_val}")