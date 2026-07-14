"""
idpromat_model.py

Simplified reimplementation of the paper's IDProMat (Section 2.3):
  - Encoder: GRU, turns an amino acid sequence into one hidden vector
  - Decoder: GRU, turns a hidden vector back into an amino acid sequence
  - MLP: learns to map one chain's hidden vector to its binding
    partner's hidden vector

Trains on a CSV with columns: pdb_id,chain_a,chain_b,seq_a,seq_b,...
(the format your batch_extract_interfaces.py produces)

Usage:
    python idpromat_model.py dataset.csv --epochs 200

Requires:
    pip install torch --break-system-packages
"""

import argparse
import csv
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = None  # set in main() based on --device argument

# 21 residue types: 20 standard amino acids + hydroxyproline (O)
VOCAB = "ACDEFGHIKLMNPQRSTVWYO"
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)


def seq_to_tensor(seq):
    """Amino acid string -> one-hot tensor, shape (1, seq_len, VOCAB_SIZE)."""
    idxs = [CHAR_TO_IDX[c] for c in seq]
    one_hot = torch.zeros(1, len(idxs), VOCAB_SIZE, device=DEVICE)
    for t, idx in enumerate(idxs):
        one_hot[0, t, idx] = 1.0
    return one_hot


def seq_to_indices(seq):
    """Amino acid string -> tensor of class indices, for cross-entropy loss."""
    return torch.tensor([CHAR_TO_IDX[c] for c in seq], device=DEVICE)


class Encoder(nn.Module):
    """Sequence -> single hidden vector."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(VOCAB_SIZE, hidden_dim, batch_first=True)

    def forward(self, x):
        _, hidden = self.gru(x)  # hidden: (1, 1, hidden_dim)
        return hidden


class Decoder(nn.Module):
    """Hidden vector -> sequence, one residue at a time.
    During training, the *actual* previous residue is fed back in
    (teacher forcing) so each step has real information to condition
    on, instead of every step seeing an identical repeated context."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        # input at each step = [context_vector ; previous residue one-hot]
        self.gru = nn.GRU(hidden_dim + VOCAB_SIZE, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(self, hidden, target_len, target_indices=None):
        """
        hidden: (1, 1, hidden_dim) - the context vector from the encoder/MLP
        target_len: how many residues to generate
        target_indices: (target_len,) ground-truth indices, used for
            teacher forcing during training. If None (prediction time),
            the decoder feeds back its own previous prediction instead.
        """
        context = hidden.permute(1, 0, 2)  # (1, 1, hidden_dim)
        h = hidden  # GRU hidden state, (1, 1, hidden_dim)

        # start token: zero vector (no "previous residue" yet)
        prev_residue = torch.zeros(1, 1, VOCAB_SIZE, device=hidden.device)

        logits_list = []
        for t in range(target_len):
            step_input = torch.cat([context, prev_residue], dim=-1)  # (1,1, hidden_dim+VOCAB_SIZE)
            out, h = self.gru(step_input, h)
            logits_t = self.out(out)  # (1, 1, VOCAB_SIZE)
            logits_list.append(logits_t)

            if target_indices is not None:
                # teacher forcing: next input = the ACTUAL residue at this position
                next_idx = target_indices[t]
            else:
                # prediction time: next input = the model's own best guess
                next_idx = logits_t.squeeze().argmax()

            prev_residue = torch.zeros(1, 1, VOCAB_SIZE, device=hidden.device)
            prev_residue[0, 0, next_idx] = 1.0

        return torch.cat(logits_list, dim=1)  # (1, target_len, VOCAB_SIZE)


class MLP(nn.Module):
    """One chain's hidden vector -> predicted partner's hidden vector."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
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
            # skip anything with characters outside our 21-letter vocab
            if all(c in CHAR_TO_IDX for c in seq_a) and all(c in CHAR_TO_IDX for c in seq_b):
                pairs.append((seq_a, seq_b))
    return pairs


def evaluate_seq2seq_accuracy(seqs, encoder, decoder):
    """Per-residue accuracy of the seq2seq module reconstructing its own input.
    This is the metric the paper reports as train/val/test accuracy (99.85/99.00/90.70%)."""
    correct, total = 0, 0
    with torch.no_grad():
        for seq in seqs:
            x = seq_to_tensor(seq)
            hidden = encoder(x)
            logits = decoder(hidden, target_len=len(seq))  # no teacher forcing at eval time
            pred_idxs = logits.squeeze(0).argmax(dim=-1).tolist()
            true_idxs = [CHAR_TO_IDX[c] for c in seq]
            correct += sum(p == t for p, t in zip(pred_idxs, true_idxs))
            total += len(seq)
    return correct / total if total > 0 else 0.0


def evaluate_partner_prediction(pairs, encoder, decoder, mlp, n_examples=3):
    """How well the FULL pipeline (encoder -> MLP -> decoder) predicts the
    actual binding partner, decoding at the CORRECT known length (fair
    comparison, since we know the true partner here)."""
    total_correct, total_residues = 0, 0
    examples = []
    with torch.no_grad():
        for seq_a, seq_b in pairs:
            h = encoder(seq_to_tensor(seq_a))
            h_pred = mlp(h)
            logits = decoder(h_pred, target_len=len(seq_b))  # correct length, not input length
            pred_idxs = logits.squeeze(0).argmax(dim=-1).tolist()
            pred_seq = "".join(VOCAB[i] for i in pred_idxs)
            true_idxs = [CHAR_TO_IDX[c] for c in seq_b]

            total_correct += sum(p == t for p, t in zip(pred_idxs, true_idxs))
            total_residues += len(seq_b)

            if len(examples) < n_examples:
                examples.append((seq_a, pred_seq, seq_b))

    accuracy = total_correct / total_residues if total_residues > 0 else 0.0
    return accuracy, examples


def split_train_val(pairs, val_fraction=0.1, seed=42):
    import random
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def train(pairs, hidden_dim=64, epochs=200, lr=1e-3, verbose_every=20):
    encoder = Encoder(hidden_dim).to(DEVICE)
    decoder = Decoder(hidden_dim).to(DEVICE)
    mlp = MLP(hidden_dim).to(DEVICE)

    seq2seq_params = list(encoder.parameters()) + list(decoder.parameters())
    seq2seq_opt = optim.Adam(seq2seq_params, lr=lr)
    mlp_opt = optim.Adam(mlp.parameters(), lr=lr)

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    # All unique sequences, for training the seq2seq reconstruction task
    all_seqs = list({s for pair in pairs for s in pair})

    for epoch in range(1, epochs + 1):
        # --- Stage 1: train seq2seq to reconstruct sequences ---
        total_seq2seq_loss = 0.0
        for seq in all_seqs:
            x = seq_to_tensor(seq)
            target = seq_to_indices(seq)

            hidden = encoder(x)
            logits = decoder(hidden, target_len=len(seq), target_indices=target)

            loss = ce_loss(logits.squeeze(0), target)
            seq2seq_opt.zero_grad()
            loss.backward()
            seq2seq_opt.step()
            total_seq2seq_loss += loss.item()

        # --- Stage 2: train MLP to map hidden(seq_a) -> hidden(seq_b) and vice versa ---
        total_mlp_loss = 0.0
        for seq_a, seq_b in pairs:
            with torch.no_grad():
                h_a = encoder(seq_to_tensor(seq_a))
                h_b = encoder(seq_to_tensor(seq_b))

            pred_b = mlp(h_a)
            pred_a = mlp(h_b)
            loss = mse_loss(pred_b, h_b) + mse_loss(pred_a, h_a)

            mlp_opt.zero_grad()
            loss.backward()
            mlp_opt.step()
            total_mlp_loss += loss.item()

        if epoch % verbose_every == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | seq2seq loss: {total_seq2seq_loss/len(all_seqs):.4f} "
                  f"| MLP loss: {total_mlp_loss/len(pairs):.4f}")

    return encoder, decoder, mlp


def predict_partner(seq, encoder, decoder, mlp):
    """Given a sequence, predict its binding partner sequence."""
    with torch.no_grad():
        h = encoder(seq_to_tensor(seq))
        h_pred = mlp(h)
        # decode at the same length as input, as a simple default
        # (no target_indices - decoder feeds back its own predictions)
        logits = decoder(h_pred, target_len=len(seq))
        idxs = logits.squeeze(0).argmax(dim=-1).tolist()
        return "".join(VOCAB[i] for i in idxs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_csv", help="CSV from batch_extract_interfaces.py")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                         help="cpu is usually FASTEST for this model, since it's many tiny "
                              "sequential steps rather than one big parallel op. Only try "
                              "mps/cuda if you rewrite the decoder to batch sequences together.")
    args = parser.parse_args()

    global DEVICE
    DEVICE = torch.device(args.device)
    print(f"Using device: {DEVICE}")

    pairs = load_dataset(args.dataset_csv)
    print(f"Loaded {len(pairs)} valid sequence pairs.")

    train_pairs, val_pairs = split_train_val(pairs, val_fraction=0.1)
    print(f"Train: {len(train_pairs)} pairs | Validation (held out, never trained on): {len(val_pairs)} pairs")

    encoder, decoder, mlp = train(train_pairs, hidden_dim=args.hidden_dim, epochs=args.epochs)

    # --- Evaluation, matching the paper's reporting style ---
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