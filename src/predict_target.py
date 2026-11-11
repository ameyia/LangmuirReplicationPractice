"""
predict_target.py

Direction B: use a trained IDProMat checkpoint to predict a binding
partner peptide for a real target sequence - the same kind of step
as the paper's GFOGER -> LRWNSYY prediction.

Key design point: the MLP predicts ONE fixed-size hidden vector for
the partner, regardless of length (that's the whole point of hidden
vectors - see the "what is hidden" discussion). But the decoder still
needs to be told how many residues to generate. So this script tries
several candidate lengths and scores each one using RECIPROCITY - the
same idea the paper used (Figure 3c): if the model's mapping is any
good, encoding the predicted peptide and running it back through the
MLP should land close to the original target's hidden vector. No
ground truth needed - this works even for a target with an unknown
true partner.

Usage:
    python predict_target.py --checkpoint_path best_model_local.pt \
        --target GFOGER --min_len 5 --max_len 20 --top_n 5

Requires:
    pip install torch --break-system-packages
    idpromat_model.py in the same folder
"""

import argparse
import torch
import torch.nn.functional as F

import idpromat_model as m

POSITIVE = set("KRH")
NEGATIVE = set("DE")
HYDROPHOBIC = set("AVLIMFWYCPO")  # rough heuristic - P/O (hydroxyproline) grouped in as structurally similar


def composition_stats(seq):
    """Rough, heuristic physicochemical summary - NOT a rigorous
    prediction, just a plausibility sanity-check alongside the
    reciprocity score."""
    pos = sum(1 for c in seq if c in POSITIVE)
    neg = sum(1 for c in seq if c in NEGATIVE)
    hydro = sum(1 for c in seq if c in HYDROPHOBIC)
    net_charge = pos - neg
    hydro_frac = hydro / len(seq) if seq else 0.0
    return net_charge, hydro_frac


def load_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    hidden_dim = ckpt["hidden_dim"]
    dropout = ckpt["dropout"]

    encoder = m.Encoder(hidden_dim, dropout=dropout).to(device)
    decoder = m.Decoder(hidden_dim, dropout=dropout).to(device)
    mlp = m.MLP(hidden_dim, dropout=dropout).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    mlp.load_state_dict(ckpt["mlp"])

    encoder.eval()
    decoder.eval()
    mlp.eval()

    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(validation partner-prediction accuracy at save time: {ckpt.get('partner_acc', 0):.2%})")
    print(f"hidden_dim={hidden_dim}, dropout={dropout}")

    return encoder, decoder, mlp


def predict_candidates(target_seq, encoder, decoder, mlp, min_len, max_len, device):
    """For a target sequence, generate one candidate peptide per
    length in [min_len, max_len], score each by reciprocity, and
    return them sorted best-first."""
    if not all(c in m.CHAR_TO_IDX for c in target_seq):
        bad = set(target_seq) - set(m.CHAR_TO_IDX)
        raise ValueError(f"Target sequence contains characters outside the trained vocabulary: {bad}")

    m.DEVICE = device
    with torch.no_grad():
        x, _, lengths = m.pad_batch([target_seq])
        h_target = encoder(x, lengths)  # (1, 1, hidden_dim)
        h_pred = mlp(h_target.squeeze(0)).unsqueeze(0)  # (1, 1, hidden_dim) - predicted partner hidden vector

        results = []
        for L in range(min_len, max_len + 1):
            logits = decoder(h_pred, max_len=L)
            idxs = logits.squeeze(0).argmax(dim=-1).tolist()
            candidate = "".join(m.VOCAB[i] for i in idxs)

            # Reciprocity: encode the candidate, run it back through the MLP,
            # see how close it lands to the ORIGINAL target's hidden vector.
            xc, _, lc = m.pad_batch([candidate])
            h_candidate = encoder(xc, lc)
            h_roundtrip = mlp(h_candidate.squeeze(0))
            reciprocity = F.cosine_similarity(h_roundtrip, h_target.squeeze(0), dim=-1).item()

            net_charge, hydro_frac = composition_stats(candidate)
            results.append({
                "length": L, "sequence": candidate, "reciprocity": reciprocity,
                "net_charge": net_charge, "hydrophobic_frac": hydro_frac,
            })

    results.sort(key=lambda r: r["reciprocity"], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict a binding partner peptide for a target sequence.")
    parser.add_argument("--checkpoint_path", default="best_model.pt")
    parser.add_argument("--target", required=True, help="Target sequence, e.g. GFOGER")
    parser.add_argument("--min_len", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    target = args.target.upper()

    print(f"Target sequence: {target}\n")
    encoder, decoder, mlp = load_checkpoint(args.checkpoint_path, device)

    print(f"\nGenerating candidates, lengths {args.min_len}-{args.max_len}...")
    results = predict_candidates(target, encoder, decoder, mlp, args.min_len, args.max_len, device)

    print(f"\nTop {args.top_n} candidates, ranked by reciprocity "
          f"(higher = more self-consistent with the target, no ground truth needed):\n")
    print(f"{'Rank':<5}{'Length':<8}{'Sequence':<25}{'Reciprocity':<13}{'Net charge':<12}{'Hydrophobic %'}")
    for i, r in enumerate(results[:args.top_n], 1):
        print(f"{i:<5}{r['length']:<8}{r['sequence']:<25}{r['reciprocity']:.4f}       "
              f"{r['net_charge']:<12}{r['hydrophobic_frac']:.1%}")

    best = results[0]
    print(f"\nBest candidate: {best['sequence']} (length {best['length']}, "
          f"reciprocity {best['reciprocity']:.4f})")
    print("\nNote: reciprocity measures internal self-consistency with the model's own")
    print("mapping, NOT a guarantee of real biological binding. Treat this as a ranked")
    print("shortlist for further validation (docking, MD, or wet-lab), the same role")
    print("the paper's own prediction played before their experimental confirmation.")


if __name__ == "__main__":
    main()
    