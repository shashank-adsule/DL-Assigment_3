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
# Google Drive checkpoint downloader
# ══════════════════════════════════════════════════════════════════════════════

# ── Fill in your actual Google Drive file IDs here ───────────────────────────
GDRIVE_FILE_IDS = {
    "checkpoints/best_model.pt" : "YOUR_BEST_MODEL_FILE_ID_HERE",
    "checkpoints/vocab.pt"      : "YOUR_VOCAB_FILE_ID_HERE",
}

def download_from_gdrive(file_id: str, dest_path: str):
    """
    Download a file from Google Drive using gdown.
    Falls back to requests if gdown is unavailable.

    Args:
        file_id   : Google Drive file ID (from the share link)
        dest_path : local path to save the file
    """
    if os.path.exists(dest_path):
        print(f"  Already exists: {dest_path}")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  Downloading {dest_path} from Google Drive …")

    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=False)

    except ImportError:
        # Fallback using requests (no gdown needed)
        import requests
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session  = requests.Session()
        response = session.get(url, stream=True)

        # Handle Google's virus-scan warning for large files
        for key, value in response.cookies.items():
            if "download_warning" in key:
                response = session.get(
                    url + f"&confirm={value}", stream=True
                )
                break

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)

    print(f"  ✓ Saved to {dest_path}")


def ensure_checkpoints():
    """
    Download best_model.pt and vocab.pt from Google Drive if not present locally.
    Call this at the start of translate() or any inference function.
    """
    for dest_path, file_id in GDRIVE_FILE_IDS.items():
        if file_id == "YOUR_BEST_MODEL_FILE_ID_HERE":
            continue   # not configured yet
        download_from_gdrive(file_id, dest_path)


# ══════════════════════════════════════════════════════════════════════════════
# Default hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = dict(
    # Model — paper base model (Table 3, section 3)
    d_model      = 512,
    num_layers   = 6,
    num_heads    = 8,
    d_ff         = 2048,
    dropout      = 0.1,
    max_len      = 200,

    # Training
    batch_size   = 64,
    num_epochs   = 35,       # more epochs = better BLEU on small dataset
    warmup_steps = 4000,
    label_smooth = 0.1,
    clip_grad    = 1.0,
    min_freq     = 2,
    seed         = 42,

    # Checkpoint averaging (paper section 6.1)
    # Save last N checkpoints and average their weights for final model
    avg_checkpoints = 5,

    # I/O
    save_path     = "checkpoints/best_model.pt",
    wandb_project = "da6401_assignment3",
    wandb_entity  = None,
)


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint Averaging  (paper section 6.1)
# ══════════════════════════════════════════════════════════════════════════════

def average_checkpoints(checkpoint_paths: list, save_path: str):
    """
    Average weights of multiple checkpoints.
    Paper: "We used a single model obtained by averaging the last 5 checkpoints"
    This consistently adds +1 to +2 BLEU over the single best checkpoint.

    Args:
        checkpoint_paths : list of .pt file paths to average
        save_path        : where to save the averaged checkpoint
    """
    print(f"\nAveraging {len(checkpoint_paths)} checkpoints …")
    avg_state = None
    ref_ckpt  = None

    for path in checkpoint_paths:
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["model_state"]
        if avg_state is None:
            avg_state = {k: v.float().clone() for k, v in state.items()}
            ref_ckpt  = ckpt
        else:
            for k in avg_state:
                avg_state[k] += state[k].float()

    # Divide by number of checkpoints
    n = len(checkpoint_paths)
    for k in avg_state:
        avg_state[k] = (avg_state[k] / n).to(
            ref_ckpt["model_state"][k].dtype
        )

    # Save averaged checkpoint (reuse metadata from most recent)
    ref_ckpt["model_state"] = avg_state
    torch.save(ref_ckpt, save_path)
    print(f"  ✓ Averaged checkpoint saved to {save_path}")


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
    ckpt_dir      = os.path.dirname(config["save_path"])
    best_val_loss = float("inf")
    recent_ckpts  = []   # track last N epoch checkpoints for averaging

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

        # ── Save best single checkpoint ───────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "src_vocab"  : src_vocab,
                "tgt_vocab"  : tgt_vocab,
                "config"     : config,
            }, config["save_path"])
            vocab_path = os.path.join(ckpt_dir, "vocab.pt")
            torch.save({"src_vocab": src_vocab, "tgt_vocab": tgt_vocab},
                       vocab_path)
            print(f"          ✓ best checkpoint saved (val_loss={val_loss:.4f})")

        # ── Save per-epoch checkpoint for averaging (last N epochs) ───────────
        # Paper section 6.1: average last 5 checkpoints for +1~2 BLEU
        epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch"      : epoch,
            "model_state": model.state_dict(),
            "src_vocab"  : src_vocab,
            "tgt_vocab"  : tgt_vocab,
            "config"     : config,
        }, epoch_path)
        recent_ckpts.append(epoch_path)

        # Keep only the last avg_checkpoints files on disk
        n_avg = config.get("avg_checkpoints", 5)
        if len(recent_ckpts) > n_avg:
            old = recent_ckpts.pop(0)
            if os.path.exists(old):
                os.remove(old)

    # ── Checkpoint averaging (paper section 6.1) ──────────────────────────────
    if len(recent_ckpts) > 1:
        avg_path = os.path.join(ckpt_dir, "averaged_model.pt")
        average_checkpoints(recent_ckpts, avg_path)

        # Evaluate averaged model — use it if better than best single ckpt
        avg_ckpt = torch.load(avg_path, map_location=device, weights_only=False)
        model.load_state_dict(avg_ckpt["model_state"])
        avg_bleu = compute_bleu(model, val_loader, tgt_vocab, device)
        print(f"  Averaged model val_bleu = {avg_bleu:.2f}")

        # Load best single model for comparison
        best_ckpt = torch.load(config["save_path"], map_location=device,
                               weights_only=False)
        model.load_state_dict(best_ckpt["model_state"])
        best_bleu = compute_bleu(model, val_loader, tgt_vocab, device)
        print(f"  Best single  val_bleu = {best_bleu:.2f}")

        # Use whichever is better as the final submission checkpoint
        if avg_bleu > best_bleu:
            print("  → Using averaged checkpoint as final model")
            import shutil
            shutil.copy(avg_path, config["save_path"])
            # Reload the better model
            model.load_state_dict(avg_ckpt["model_state"])
        else:
            print("  → Keeping best single checkpoint as final model")

        wandb.log({"val/bleu_averaged": avg_bleu, "val/bleu_best_single": best_bleu})

    # ── Final test-set BLEU ───────────────────────────────────────────────────
    print("\nEvaluating on test set …")
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab
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

    # Download from Google Drive if not present locally
    ensure_checkpoints()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt      = torch.load(checkpoint_path, map_location=device, weights_only=False)
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