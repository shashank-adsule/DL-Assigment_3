"""
train.py
--------
Training pipeline for the Transformer NMT model.

Run training:
    python train.py

Override hyperparams:
    python train.py --d_model 512 --num_layers 6 --num_epochs 30

Translate after training:
    python train.py --mode translate \
                    --checkpoint checkpoints/best_model.pt \
                    --sentence "Ein Hund spielt im Park."
"""

import os
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from dataset      import get_dataset, PAD_IDX, BOS_IDX, EOS_IDX
from model        import Transformer
from lr_scheduler import get_optimizer_and_scheduler


# ══════════════════════════════════════════════════════════════════════════════
# Default hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = dict(
    # Model
    d_model      = 256,
    num_layers   = 3,
    num_heads    = 8,
    d_ff         = 512,
    dropout      = 0.1,
    max_len      = 200,

    # Training
    batch_size   = 128,
    num_epochs   = 20,
    warmup_steps = 4000,
    label_smooth = 0.1,
    clip_grad    = 1.0,
    min_freq     = 2,
    seed         = 42,

    # I/O
    save_path     = "checkpoints/best_model.pt",
    wandb_project = "da6401_assignment3",
    wandb_entity  = None,
)


# ══════════════════════════════════════════════════════════════════════════════
# Label-Smoothing Loss
# ══════════════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing (ε_ls = 0.1 as required by Task 3).

    Instead of a hard one-hot target, we assign:
        (1 - ε_ls)               → correct token
        ε_ls / (vocab_size - 2)  → every other non-pad token
        0                        → <pad> token

    This regularises the model and prevents over-confident predictions.

    Args:
        vocab_size : target vocabulary size
        pad_idx    : index of <pad> token (excluded from loss)
        smoothing  : ε_ls  (default 0.1)
    """

    def __init__(self, vocab_size: int, pad_idx: int = 0,
                 smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        """
        Args:
            logits  : (N, vocab_size)  — flattened raw model output
            targets : (N,)             — flattened gold token ids
        Returns:
            loss : scalar
        """
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs,
                                          self.smoothing / (self.vocab_size - 2))
            smooth_dist[:, self.pad_idx] = 0.0
            smooth_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
            non_pad = (targets != self.pad_idx)
            smooth_dist[~non_pad] = 0.0

        loss     = -(smooth_dist * log_probs).sum(dim=-1)
        n_tokens = non_pad.sum().float().clamp(min=1)
        return loss.sum() / n_tokens


# ══════════════════════════════════════════════════════════════════════════════
# BLEU evaluation
# ══════════════════════════════════════════════════════════════════════════════

def compute_bleu(model, data_loader, tgt_vocab, device, max_len: int = 100):
    """
    Corpus-level BLEU score via greedy decoding.
    Uses the `evaluate` library (sacrebleu backend) when available.

    Returns:
        bleu_score : float in [0, 100]
    """
    try:
        import evaluate as hf_evaluate
        bleu_metric = hf_evaluate.load("bleu")
        use_hf = True
    except Exception:
        use_hf = False

    model.eval()
    predictions, references = [], []

    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(data_loader, desc="BLEU", leave=False):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            for i in range(src_batch.size(0)):
                src      = src_batch[i].unsqueeze(0)
                pred_ids = model.infer(src, BOS_IDX, EOS_IDX, PAD_IDX, max_len)
                pred_ids = [t for t in pred_ids
                            if t not in (BOS_IDX, EOS_IDX, PAD_IDX)]

                ref_ids  = tgt_batch[i].tolist()
                ref_ids  = [t for t in ref_ids
                            if t not in (BOS_IDX, EOS_IDX, PAD_IDX)]

                predictions.append(" ".join(tgt_vocab.decode(pred_ids)))
                references.append([" ".join(tgt_vocab.decode(ref_ids))])

    if use_hf:
        result = bleu_metric.compute(predictions=predictions,
                                     references=references)
        return result["bleu"] * 100

    # Fallback: simple 4-gram precision
    from collections import Counter
    def ngrams(toks, n):
        return Counter(tuple(toks[i:i+n]) for i in range(len(toks) - n + 1))
    total_match, total_pred = 0, 0
    for pred, refs in zip(predictions, references):
        p_tok = pred.split(); r_tok = refs[0].split()
        p4 = ngrams(p_tok, 4); r4 = ngrams(r_tok, 4)
        for ng, cnt in p4.items():
            total_match += min(cnt, r4.get(ng, 0))
        total_pred += max(len(p_tok) - 3, 0)
    return (total_match / max(total_pred, 1)) * 100


# ══════════════════════════════════════════════════════════════════════════════
# Training / evaluation  (also imported by experiments.py)
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler, criterion, device,
                clip_grad: float = 1.0):
    """One full training pass. Returns average loss per non-pad token."""
    model.train()
    total_loss, n_batches = 0.0, 0

    for src, tgt in tqdm(loader, desc="Train", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input  = tgt[:, :-1]   # input:  <bos> w1 w2 … wN
        tgt_output = tgt[:, 1:]    # target: w1 w2 … wN <eos>

        logits = model(src, tgt_input)
        B, T, V = logits.size()

        loss = criterion(logits.contiguous().view(-1, V),
                         tgt_output.contiguous().view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        scheduler.step()    # Noam: update LR before optimizer.step()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def evaluate_loss(model, loader, criterion, device):
    """Validation loss with no gradients."""
    model.eval()
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)
            B, T, V = logits.size()

            loss = criterion(logits.contiguous().view(-1, V),
                             tgt_output.contiguous().view(-1))
            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Main training script
# ══════════════════════════════════════════════════════════════════════════════

def train(config: dict):
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wandb.init(project=config["wandb_project"],
               entity=config.get("wandb_entity"),
               config=config, name="main_training")

    # Data
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataset(
        batch_size=config["batch_size"],
        min_freq=config["min_freq"],
        max_len=config["max_len"],
    )

    # Model
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        max_len=config["max_len"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    criterion            = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX,
                                              config["label_smooth"])
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, config["d_model"], config["warmup_steps"]
    )

    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, config["num_epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 criterion, device, config["clip_grad"])
        val_loss   = evaluate_loss(model, val_loader, criterion, device)
        train_ppl  = math.exp(min(train_loss, 100))
        val_ppl    = math.exp(min(val_loss,   100))

        print(f"Epoch {epoch:02d} | "
              f"train_loss={train_loss:.4f} ppl={train_ppl:.1f} | "
              f"val_loss={val_loss:.4f} ppl={val_ppl:.1f} | "
              f"lr={scheduler.current_lr:.6f}")

        log = {
            "epoch"           : epoch,
            "train/loss"      : train_loss,
            "train/perplexity": train_ppl,
            "val/loss"        : val_loss,
            "val/perplexity"  : val_ppl,
            "lr"              : scheduler.current_lr,
            "step"            : scheduler.current_step,
        }

        # BLEU every 5 epochs (greedy decoding over full val set is slow)
        if epoch % 5 == 0 or epoch == config["num_epochs"]:
            val_bleu = compute_bleu(model, val_loader, tgt_vocab, device)
            log["val/bleu"] = val_bleu
            print(f"          val_bleu={val_bleu:.2f}")

        wandb.log(log)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "src_vocab"  : src_vocab,
                "tgt_vocab"  : tgt_vocab,
                "config"     : config,
            }, config["save_path"])
            print(f"          ✓ checkpoint saved (val_loss={val_loss:.4f})")

    # Final test-set BLEU from best checkpoint
    print("\nEvaluating on test set …")
    ckpt = torch.load(config["save_path"], map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_bleu = compute_bleu(model, test_loader, tgt_vocab, device)
    print(f"Test BLEU: {test_bleu:.2f}")
    wandb.log({"test/bleu": test_bleu})
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

def translate(checkpoint_path: str, sentence: str, max_len: int = 100):
    """Load a checkpoint and translate one German sentence to English."""
    from dataset import load_spacy_models, tokenize_de

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt      = torch.load(checkpoint_path, map_location=device)
    cfg       = ckpt["config"]
    src_vocab = ckpt["src_vocab"]
    tgt_vocab = ckpt["tgt_vocab"]

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    spacy_de, _ = load_spacy_models()
    tokens      = tokenize_de(sentence, spacy_de)
    ids         = [BOS_IDX] + src_vocab.encode(tokens) + [EOS_IDX]
    src         = torch.tensor([ids], dtype=torch.long, device=device)

    pred_ids    = model.infer(src, BOS_IDX, EOS_IDX, PAD_IDX, max_len)
    pred_ids    = [t for t in pred_ids if t not in (EOS_IDX, PAD_IDX)]
    translation = " ".join(tgt_vocab.decode(pred_ids))

    print(f"DE: {sentence}")
    print(f"EN: {translation}")
    return translation


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer DE→EN")
    parser.add_argument("--mode", default="train",
                        choices=["train", "translate"])

    for k, v in DEFAULT_CONFIG.items():
        t = type(v) if v is not None else str
        parser.add_argument(f"--{k}", type=t, default=v)

    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sentence",   default=None)

    args   = parser.parse_args()
    config = {k: getattr(args, k) for k in DEFAULT_CONFIG}

    if args.mode == "train":
        train(config)
    elif args.mode == "translate":
        ckpt = args.checkpoint or config["save_path"]
        sent = args.sentence   or "Ein Hund spielt im Park."
        translate(ckpt, sent)