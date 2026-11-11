"""
sanity_check_reciprocity.py

Two checks on whether the model's "reciprocity" score means anything.

CHECK A - does reciprocity separate real partners from junk?
  For each held-out validation pair (seq_a, seq_b), score how well several
  candidate partners "round-trip" back to seq_a:
      reciprocity(candidate) = cosine( MLP(Encoder(candidate)), Encoder(seq_a) )
  Candidates compared, all the same length as the true partner:
      TRUE      the real partner from the PDB
      PREDICTED what the model generates for seq_a
      SCRAMBLED the true partner with its residues shuffled (same composition)
      RANDOM    random peptides drawn from the dataset's residue frequencies
  If reciprocity is meaningful, TRUE should clearly beat RANDOM/SCRAMBLED.
  If they are about equal, the score is not evidence of binding.

CHECK B - mode collapse
  Run ~10 unrelated targets through the model, take each top candidate, and
  see how much they look alike (shared prefixes). If unrelated targets give
  near-identical outputs, the model is ignoring its input.

Usage (from the src/ folder):
    python sanity_check_reciprocity.py \
        --checkpoint_path ../outputs/models/best_model_local.pt \
        --dataset ../data/extracted/dataset.csv --device cpu
"""

import argparse
import random
from collections import Counter

import torch
import torch.nn.functional as F

import idpromat_model as m


def load_models(path, device):
    ckpt = torch.load(path, map_location=device)
    hd, dp = ckpt["hidden_dim"], ckpt.get("dropout", 0.0)
    enc, dec, mlp = m.Encoder(hd, dp).to(device), m.Decoder(hd, dp).to(device), m.MLP(hd, dp).to(device)
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    mlp.load_state_dict(ckpt["mlp"])
    for net in (enc, dec, mlp):
        net.eval()
    return enc, dec, mlp, ckpt


@torch.no_grad()
def encode(seqs, enc):
    x, _, lengths = m.pad_batch(seqs)
    return enc(x, lengths).squeeze(0)  # (batch, hidden)


@torch.no_grad()
def reciprocity(candidates, h_target, enc, mlp):
    """Cosine of the round-trip vector vs the target's vector, one score per candidate."""
    h_c = encode(candidates, enc)
    h_round = mlp(h_c)
    return F.cosine_similarity(h_round, h_target.expand_as(h_round), dim=-1).tolist()


@torch.no_grad()
def generate(target, enc, dec, mlp, length):
    h = encode([target], enc)
    h_pred = mlp(h).unsqueeze(0)
    logits = dec(h_pred, max_len=length)
    return "".join(m.VOCAB[i] for i in logits[0].argmax(-1).tolist())


def mean(xs):
    return sum(xs) / len(xs)


def check_a(val_pairs, all_residues, enc, dec, mlp, n_random, rng):
    rows = []
    for a, b in val_pairs:
        L = len(b)
        h_a = encode([a], enc)
        pred = generate(a, enc, dec, mlp, L)
        scr = list(b)
        rng.shuffle(scr)
        scr = "".join(scr)
        rands = ["".join(rng.choice(all_residues) for _ in range(L)) for _ in range(n_random)]
        s_true, s_pred, s_scr = (reciprocity([x], h_a, enc, mlp)[0] for x in (b, pred, scr))
        s_rand = reciprocity(rands, h_a, enc, mlp)
        rows.append({
            "true": s_true, "pred": s_pred, "scr": s_scr, "rand": mean(s_rand),
            "true_beats_rand": mean([s_true > r for r in s_rand]),
            "pred_beats_rand": mean([s_pred > r for r in s_rand]),
        })
    print("=" * 70)
    print(f"CHECK A: reciprocity on {len(rows)} held-out validation pairs")
    print("=" * 70)
    print(f"  mean reciprocity  TRUE partner       : {mean([r['true'] for r in rows]):.3f}")
    print(f"  mean reciprocity  MODEL-PREDICTED    : {mean([r['pred'] for r in rows]):.3f}")
    print(f"  mean reciprocity  SCRAMBLED true     : {mean([r['scr'] for r in rows]):.3f}")
    print(f"  mean reciprocity  RANDOM peptides    : {mean([r['rand'] for r in rows]):.3f}")
    print(f"  P(TRUE      > a random peptide)      : {mean([r['true_beats_rand'] for r in rows]):.1%}")
    print(f"  P(PREDICTED > a random peptide)      : {mean([r['pred_beats_rand'] for r in rows]):.1%}")
    print("  (50% = no better than chance; ~100% = score clearly favors it)\n")
    t, p = mean([r["true_beats_rand"] for r in rows]), mean([r["pred_beats_rand"] for r in rows])
    if t < 0.6:
        print("  READ: reciprocity barely separates TRUE from random -> weak evidence of binding.")
    else:
        print("  READ: reciprocity does favor TRUE partners over random peptides.")
    if p < 0.6:
        print("  READ: model-generated candidates are NOT scoring above random -> ranking by")
        print("        reciprocity is not selecting anything meaningful.")
    else:
        print("  READ: model-generated candidates score above random peptides.")
    print()


def check_b(targets, enc, dec, mlp, min_len, max_len):
    print("=" * 70)
    print(f"CHECK B: mode-collapse test on {len(targets)} unrelated targets")
    print("=" * 70)
    tops = []
    for t in targets:
        full = generate(t, enc, dec, mlp, max_len)  # greedy: shorter candidates are prefixes
        cands = [full[:L] for L in range(min_len, max_len + 1)]
        scores = reciprocity(cands, encode([t], enc), enc, mlp)
        best = max(range(len(cands)), key=lambda i: scores[i])
        tops.append(cands[best])
        print(f"  target {t[:28]:<28} -> {cands[best]:<{max_len}} (recip {scores[best]:.3f}, len {len(cands[best])})")

    def lcp(x, y):
        n = 0
        for c, d in zip(x, y):
            if c != d:
                break
            n += 1
        return n

    k = 4
    prefixes = Counter(t[:k] for t in tops)
    top_prefix, top_count = prefixes.most_common(1)[0]
    pair_lcp = [lcp(x, y) for i, x in enumerate(tops) for y in tops[i + 1:]]
    print(f"\n  distinct top candidates          : {len(set(tops))} / {len(tops)}")
    print(f"  most common first-{k} residues     : '{top_prefix}' in {top_count}/{len(tops)} outputs")
    print(f"  mean shared prefix, any two      : {mean(pair_lcp):.1f} residues")
    comp = Counter("".join(tops))
    n = sum(comp.values())
    print("  residues across all outputs      : " + ", ".join(f"{a}:{c / n:.0%}" for a, c in comp.most_common(5)))
    if top_count / len(tops) >= 0.5 or mean(pair_lcp) >= 3:
        print("\n  READ: outputs share a prefix across unrelated targets -> likely mode collapse")
        print("        (the decoder leans on its own habits more than on the target).")
    else:
        print("\n  READ: outputs differ across targets -> no strong sign of mode collapse.")
    print()


DEFAULT_TARGETS = [
    "GPOGPOGPOGPOGPO",            # collagen-like repeat
    "GFOGERGFOGERGFOGER",         # collagen-like motif
    "RGDSPASSKP",                 # integrin-binding style
    "KKKKRRRKKK",                 # very basic
    "DDEEDDEEDD",                 # very acidic
    "LLVVIILLAAV",                # hydrophobic
    "SSTTGGSSNNQQ",               # polar / small
    "MKTAYIAKQRQISFVKSHFSRQ",     # mixed, protein-like
    "WWYYFFHHWY",                 # aromatic
    "ACDEFGHIKLMNPQRSTVWY",       # one of each residue
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--n_random", type=int, default=50, help="random peptides per validation pair")
    ap.add_argument("--max_pairs", type=int, default=0, help="limit validation pairs (0 = all)")
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m.DEVICE = torch.device(args.device)
    enc, dec, mlp, ckpt = load_models(args.checkpoint_path, m.DEVICE)
    print(f"Loaded {args.checkpoint_path} (epoch {ckpt.get('epoch')}, "
          f"saved val partner acc {ckpt.get('partner_acc', float('nan')):.2%})\n")

    pairs = m.load_dataset(args.dataset)
    _, val_pairs = m.split_train_val(pairs, 0.1, seed=42)  # same split as training
    if args.max_pairs:
        val_pairs = val_pairs[:args.max_pairs]
    all_residues = [c for a, b in pairs for c in a + b]

    rng = random.Random(args.seed)
    check_a(val_pairs, all_residues, enc, dec, mlp, args.n_random, rng)
    check_b(DEFAULT_TARGETS, enc, dec, mlp, args.min_len, args.max_len)


if __name__ == "__main__":
    main()