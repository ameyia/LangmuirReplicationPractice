"""
dedupe_dataset.py

1) LEAKAGE CHECK: how similar are your validation pairs to your training pairs?
   Uses the exact same split as training (seed 42, 10%).
2) DEDUPE: groups near-duplicate pairs into clusters, keeps one per cluster,
   and writes a cleaned CSV you can retrain on.

Two pairs are "near-duplicates" if BOTH chains are similar (identity >= threshold),
checking either chain order (A,B) or (B,A), since chain labels are arbitrary.
Similarity of two sequences = difflib ratio (0 to 1, 1 = identical).

Usage (from src/):
    python dedupe_dataset.py ../data/extracted/dataset.csv --threshold 0.8
Output:
    ../data/extracted/dataset_dedup.csv   (same columns as dataset.csv)
    ../data/extracted/dataset_clusters.csv (which pair went into which cluster)

No torch needed. Pure Python, takes about a minute or two on ~2400 pairs.
"""

import argparse
import csv
import random
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

VOCAB = set("ACDEFGHIKLMNPQRSTVWYO")


def seq_sim(x, y, cache):
    if x == y:
        return 1.0
    key = (x, y) if x < y else (y, x)
    if key in cache:
        return cache[key]
    # cheap length filter: ratio can't exceed 2*min/(len x + len y)
    if 2 * min(len(x), len(y)) / (len(x) + len(y)) < 0.5:
        cache[key] = 0.0
        return 0.0
    sm = SequenceMatcher(None, x, y, autojunk=False)
    if sm.real_quick_ratio() < 0.5 or sm.quick_ratio() < 0.5:
        r = 0.0
    else:
        r = sm.ratio()
    cache[key] = r
    return r


def pair_sim(p, q, cache):
    """Similarity of two (seq_a, seq_b) pairs: both chains must match, either order."""
    a1, b1 = p
    a2, b2 = q
    straight = min(seq_sim(a1, a2, cache), seq_sim(b1, b2, cache))
    if straight == 1.0:
        return 1.0
    swapped = min(seq_sim(a1, b2, cache), seq_sim(b1, a2, cache))
    return max(straight, swapped)


def load_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if all(c in VOCAB for c in r["seq_a"]) and all(c in VOCAB for c in r["seq_b"]):
                rows.append(r)
    return rows


def split_like_training(rows, val_fraction=0.1, seed=42):
    """Mirrors idpromat_model.split_train_val so we audit the REAL split."""
    pairs = [(r["seq_a"], r["seq_b"]) for r in rows]
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def leakage_report(train, val, cache):
    print("=" * 66)
    print(f"LEAKAGE CHECK: {len(val)} validation pairs vs {len(train)} training pairs")
    print("=" * 66)
    best = []
    for v in val:
        best.append(max(pair_sim(v, t, cache) for t in train))
    n = len(best)
    for thr in (1.0, 0.9, 0.8, 0.7, 0.6):
        k = sum(b >= thr for b in best)
        label = "identical" if thr == 1.0 else f">= {thr:.0%} similar"
        print(f"  val pairs with a train pair that is {label:<16}: {k:4d} / {n}  ({k / n:.1%})")
    print(f"  median closest-train-neighbour similarity       : {sorted(best)[n // 2]:.2f}")
    print()
    return best


def cluster(pairs, threshold, cache):
    """Union-find over pairs; link any two with pair_sim >= threshold."""
    n = len(pairs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if pair_sim(pairs[i], pairs[j], cache) >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    return [find(i) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_csv")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="pairs at least this similar (both chains) count as duplicates")
    ap.add_argument("--out", default=None, help="deduped CSV path (default: next to input)")
    args = ap.parse_args()

    src = Path(args.dataset_csv)
    out = Path(args.out) if args.out else src.with_name(src.stem + "_dedup.csv")
    clusters_out = src.with_name(src.stem + "_clusters.csv")

    rows = load_rows(src)
    print(f"Loaded {len(rows)} valid pairs from {src}\n")
    cache = {}

    # --- 1) leakage in the CURRENT split ---
    train, val = split_like_training(rows)
    leakage_report(train, val, cache)

    # --- 2) dedupe whole dataset ---
    pairs = [(r["seq_a"], r["seq_b"]) for r in rows]
    print(f"Clustering at threshold {args.threshold:.2f} (this is the slow part)...")
    labels = cluster(pairs, args.threshold, cache)
    sizes = Counter(labels)

    kept, seen = [], set()
    for r, lab in zip(rows, labels):
        if lab not in seen:
            seen.add(lab)
            kept.append(r)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(kept)
    with open(clusters_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdb_id", "chain_a", "chain_b", "cluster_id", "cluster_size"])
        for r, lab in zip(rows, labels):
            w.writerow([r["pdb_id"], r["chain_a"], r["chain_b"], lab, sizes[lab]])

    big = sizes.most_common(5)
    print(f"\n  pairs before : {len(rows)}")
    print(f"  clusters     : {len(sizes)}  (= pairs kept after dedupe)")
    print(f"  removed      : {len(rows) - len(kept)}  ({(len(rows) - len(kept)) / len(rows):.1%})")
    print(f"  largest clusters (size): {[s for _, s in big]}")
    for lab, s in big[:3]:
        ex = next(r for r, l in zip(rows, labels) if l == lab)
        ids = sorted({r['pdb_id'] for r, l in zip(rows, labels) if l == lab})
        print(f"    cluster of {s}: e.g. {ex['seq_a'][:20]} / {ex['seq_b'][:20]}   PDBs: {', '.join(ids[:8])}{'...' if len(ids) > 8 else ''}")
    print(f"\nWrote: {out}\nWrote: {clusters_out}")

    # --- 3) leakage after dedupe ---
    kept_pairs = [(r["seq_a"], r["seq_b"]) for r in kept]
    train2, val2 = split_like_training(kept)
    print("\nAfter dedupe (same split recipe on the cleaned data):")
    leakage_report(train2, val2, cache)
    print("Next: retrain on the _dedup.csv file, then re-run sanity_check_reciprocity.py with it.")


if __name__ == "__main__":
    main()