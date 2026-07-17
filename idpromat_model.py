"""
idpromat_model.py

Batched reimplementation of the paper's IDProMat (Section 2.3):
  - Encoder: GRU, turns a BATCH of amino acid sequences into hidden vectors
  - Decoder: GRU, turns a BATCH of hidden vectors back into sequences
  - MLP: learns to map one chain's hidden vector to its binding
    partner's hidden vector

Unlike the earlier version, this processes many sequences at once
(padded to a common length within each batch, with padding masked
out of the loss) instead of one sequence at a time. This is what
actually makes a GPU worth using - the earlier one-at-a-time version
had too much per-step overhead for a GPU to help.

Usage:
    python idpromat_model.py dataset.csv --epochs 300 --batch_size 32

Requires:
    pip install torch --break-system-packages
"""

import argparse
import csv
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = None  # set in main() based on --device argument

# 21 residue types: 20 standard amino acids + hydroxyproline (O)
VOCAB = "ACDEFGHIKLMNPQRSTVWYO"
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)


def pad_batch(seqs):
    """Turn a list of amino acid strings (possibly different lengths) into:
      - x: one-hot tensor (batch, max_len, VOCAB_SIZE), zero-padded
      - idx: class-index tensor (batch, max_len), padded with 0 (arbitrary -
        padded positions are masked out of any loss computation, so the
        padding value itself doesn't matter)
      - lengths: (batch,) tensor of each sequence's TRUE length
    """
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    batch = len(seqs)
    x = torch.zeros(batch, max_len, VOCAB_SIZE, device=DEVICE)
    idx = torch.zeros(batch, max_len, dtype=torch.long, device=DEVICE)
    for i, s in enumerate(seqs):
        for t, c in enumerate(s):
            ci = CHAR_TO_IDX[c]
            x[i, t, ci] = 1.0
            idx[i, t] = ci
    lengths_t = torch.tensor(lengths, dtype=torch.long)  # CPU, required by pack_padded_sequence
    return x, idx, lengths_t


def masked_ce_loss(logits, targets, lengths):
    """Cross-entropy loss that ignores padded positions.
    logits: (batch, max_len, VOCAB_SIZE), targets: (batch, max_len), lengths: (batch,)"""
    batch, max_len, vocab = logits.shape
    ce = nn.functional.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction='none')
    ce = ce.view(batch, max_len)
    mask = (torch.arange(max_len, device=logits.device).unsqueeze(0)
            < lengths.unsqueeze(1).to(logits.device)).float()
    return (ce * mask).sum() / mask.sum()


def format_duration(seconds):
    """Format seconds as H:MM:SS or M:SS, whichever is more readable."""
    seconds = int(seconds)
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    if hrs > 0:
        return f"{hrs}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


class Encoder(nn.Module):
    """Batch of sequences -> batch of hidden vectors."""
    def __init__(self, hidden_dim=64, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(VOCAB_SIZE, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        """x: (batch, max_len, VOCAB_SIZE), lengths: (batch,) CPU tensor of true lengths."""
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)  # hidden: (1, batch, hidden_dim) - padding correctly ignored
        hidden = self.dropout(hidden)
        return hidden


class Decoder(nn.Module):
    """Batch of hidden vectors -> batch of sequences (all decoded to the
    same max_len; callers slice each example down to its own true length).
    Teacher forcing: if target_indices is given, the ACTUAL previous
    residue is fed at each step. If not, the decoder feeds back its own
    previous prediction (used at real prediction time)."""
    def __init__(self, hidden_dim=64, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(hidden_dim + VOCAB_SIZE, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(self, hidden, max_len, target_indices=None):
        """hidden: (1, batch, hidden_dim). Returns logits (batch, max_len, VOCAB_SIZE)."""
        batch = hidden.size(1)
        context = hidden.permute(1, 0, 2)  # (batch, 1, hidden_dim)
        h = hidden  # GRU hidden state: (1, batch, hidden_dim)

        prev_residue = torch.zeros(batch, 1, VOCAB_SIZE, device=hidden.device)
        logits_list = []
        for t in range(max_len):
            step_input = torch.cat([context, prev_residue], dim=-1)  # (batch, 1, hidden_dim+VOCAB_SIZE)
            out, h = self.gru(step_input, h)
            out = self.dropout(out)
            logits_t = self.out(out)  # (batch, 1, VOCAB_SIZE)
            logits_list.append(logits_t)

            if target_indices is not None:
                next_idx = target_indices[:, t]  # (batch,) - actual residue at this position
            else:
                next_idx = logits_t.squeeze(1).argmax(dim=-1)  # (batch,) - model's own guess

            prev_residue = torch.zeros(batch, 1, VOCAB_SIZE, device=hidden.device)
            prev_residue[torch.arange(batch, device=hidden.device), 0, next_idx] = 1.0

        return torch.cat(logits_list, dim=1)  # (batch, max_len, VOCAB_SIZE)


class MLP(nn.Module):
    """Batch of one chain's hidden vectors -> batch of predicted partner hidden vectors."""
    def __init__(self, hidden_dim=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, h):
        return self.net(h)


def load_dataset(csv_path):
    pairs = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_a, seq_b = row["seq_a"], row["seq_b"]
            if all(c in CHAR_TO_IDX for c in seq_a) and all(c in CHAR_TO_IDX for c in seq_b):
                pairs.append((seq_a, seq_b))
    return pairs


def split_train_val(pairs, val_fraction=0.1, seed=42):
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def evaluate_seq2seq_accuracy(seqs, encoder, decoder, batch_size=64):
    """Per-residue accuracy of the seq2seq module reconstructing its own input."""
    encoder.eval()
    decoder.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size]
            x, idx, lengths = pad_batch(batch)
            hidden = encoder(x, lengths)
            logits = decoder(hidden, max_len=lengths.max().item())  # no teacher forcing at eval time
            preds = logits.argmax(dim=-1)  # (batch, max_len)
            for b, L in enumerate(lengths.tolist()):
                p = preds[b, :L].tolist()
                t = idx[b, :L].tolist()
                correct += sum(pp == tt for pp, tt in zip(p, t))
                total += L
    return correct / total if total > 0 else 0.0


def evaluate_partner_prediction(pairs, encoder, decoder, mlp, n_examples=3, batch_size=32):
    """Full pipeline (encoder -> MLP -> decoder) partner-prediction accuracy,
    decoding at each example's CORRECT known length."""
    encoder.eval()
    decoder.eval()
    mlp.eval()
    total_correct, total_residues = 0, 0
    examples = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            seq_a_list = [p[0] for p in batch_pairs]
            seq_b_list = [p[1] for p in batch_pairs]

            xa, _, la = pad_batch(seq_a_list)
            h = encoder(xa, la)
            h_pred = mlp(h.squeeze(0)).unsqueeze(0)  # (1, batch, hidden_dim)

            lb_list = [len(s) for s in seq_b_list]
            max_len_b = max(lb_list)
            logits = decoder(h_pred, max_len=max_len_b)
            preds = logits.argmax(dim=-1)  # (batch, max_len_b)

            for b, (seq_a, seq_b) in enumerate(batch_pairs):
                L = len(seq_b)
                pred_idxs = preds[b, :L].tolist()
                pred_seq = "".join(VOCAB[i] for i in pred_idxs)
                true_idxs = [CHAR_TO_IDX[c] for c in seq_b]
                total_correct += sum(p == t for p, t in zip(pred_idxs, true_idxs))
                total_residues += L
                if len(examples) < n_examples:
                    examples.append((seq_a, pred_seq, seq_b))

    accuracy = total_correct / total_residues if total_residues > 0 else 0.0
    return accuracy, examples


def train(pairs, hidden_dim=64, epochs=200, lr=1e-3, dropout=0.0, batch_size=32, verbose_every=20):
    encoder = Encoder(hidden_dim, dropout=dropout).to(DEVICE)
    decoder = Decoder(hidden_dim, dropout=dropout).to(DEVICE)
    mlp = MLP(hidden_dim, dropout=dropout).to(DEVICE)

    seq2seq_params = list(encoder.parameters()) + list(decoder.parameters())
    seq2seq_opt = optim.Adam(seq2seq_params, lr=lr)
    mlp_opt = optim.Adam(mlp.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    all_seqs = list({s for pair in pairs for s in pair})

    encoder.train()
    decoder.train()
    mlp.train()

    training_start = time.time()
    last_checkpoint_time = training_start

    for epoch in range(1, epochs + 1):
        # --- Stage 1: seq2seq reconstruction, batched ---
        random.shuffle(all_seqs)
        total_seq2seq_loss, n_batches = 0.0, 0
        for i in range(0, len(all_seqs), batch_size):
            batch_seqs = all_seqs[i:i + batch_size]
            x, idx, lengths = pad_batch(batch_seqs)
            hidden = encoder(x, lengths)
            logits = decoder(hidden, max_len=lengths.max().item(), target_indices=idx)
            loss = masked_ce_loss(logits, idx, lengths)

            seq2seq_opt.zero_grad()
            loss.backward()
            seq2seq_opt.step()
            total_seq2seq_loss += loss.item()
            n_batches += 1

        # --- Stage 2: MLP partner mapping, batched ---
        random.shuffle(pairs)
        total_mlp_loss, n_batches2 = 0.0, 0
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            seq_a_list = [p[0] for p in batch_pairs]
            seq_b_list = [p[1] for p in batch_pairs]

            xa, _, la = pad_batch(seq_a_list)
            xb, _, lb = pad_batch(seq_b_list)
            with torch.no_grad():
                h_a = encoder(xa, la).squeeze(0)  # (batch, hidden_dim)
                h_b = encoder(xb, lb).squeeze(0)

            pred_b = mlp(h_a)
            pred_a = mlp(h_b)
            loss = mse_loss(pred_b, h_b) + mse_loss(pred_a, h_a)

            mlp_opt.zero_grad()
            loss.backward()
            mlp_opt.step()
            total_mlp_loss += loss.item()
            n_batches2 += 1

        if epoch % verbose_every == 0 or epoch == 1:
            now = time.time()
            elapsed_total = now - training_start
            elapsed_since_last = now - last_checkpoint_time
            avg_per_epoch = elapsed_total / epoch
            eta = avg_per_epoch * (epochs - epoch)

            print(f"Epoch {epoch:4d} | seq2seq loss: {total_seq2seq_loss/n_batches:.4f} "
                  f"| MLP loss: {total_mlp_loss/n_batches2:.4f} "
                  f"| this block: {format_duration(elapsed_since_last)} "
                  f"| elapsed: {format_duration(elapsed_total)} "
                  f"| ETA: {format_duration(eta)}")
            last_checkpoint_time = now

    total_time = time.time() - training_start
    print(f"\nTraining finished in {format_duration(total_time)} "
          f"({total_time/epochs:.2f} sec/epoch average)")

    return encoder, decoder, mlp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_csv", help="CSV from batch_extract_interfaces.py")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2,
                         help="Dropout rate (0.0-0.5 typical). 0.0 disables it entirely.")
    parser.add_argument("--batch_size", type=int, default=32,
                         help="Sequences processed together per step. Bigger = faster on GPU "
                              "(up to a point), but uses more memory.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                         help="Now that training is batched, cuda/mps should actually help "
                              "if you have a real GPU available.")
    args = parser.parse_args()

    global DEVICE
    DEVICE = torch.device(args.device)
    print(f"Using device: {DEVICE}")

    pairs = load_dataset(args.dataset_csv)
    print(f"Loaded {len(pairs)} valid sequence pairs.")

    train_pairs, val_pairs = split_train_val(pairs, val_fraction=0.1)
    print(f"Train: {len(train_pairs)} pairs | Validation (held out, never trained on): {len(val_pairs)} pairs")

    encoder, decoder, mlp = train(
        train_pairs, hidden_dim=args.hidden_dim, epochs=args.epochs,
        dropout=args.dropout, batch_size=args.batch_size,
    )

    train_seqs = list({s for pair in train_pairs for s in pair})
    val_seqs = list({s for pair in val_pairs for s in pair})

    train_acc = evaluate_seq2seq_accuracy(train_seqs, encoder, decoder)
    val_acc = evaluate_seq2seq_accuracy(val_seqs, encoder, decoder)
    print(f"\nSeq2seq reconstruction accuracy - train: {train_acc:.2%} | validation: {val_acc:.2%}")

    partner_acc, examples = evaluate_partner_prediction(val_pairs, encoder, decoder, mlp)
    print(f"Partner-prediction accuracy (held-out, correct length used): {partner_acc:.2%}")

    print("\nExample predictions on held-out validation pairs (never seen during training):")
    for seq_a, predicted, actual in examples:
        print(f"  input:     {seq_a}")
        print(f"  predicted: {predicted}")
        print(f"  actual:    {actual}\n")


if __name__ == "__main__":
    main()