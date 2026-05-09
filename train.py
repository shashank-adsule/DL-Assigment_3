"""
train.py
--------
Training pipeline, label-smoothing loss, BLEU evaluation, W&B logging,
and all five ablation experiments for the W&B report.

Run training:
    python train.py

Run a specific experiment:
    python train.py --mode exp_noam_vs_fixed
    python train.py --mode exp_scaling_factor
    python train.py --mode exp_attn_rollout
    python train.py --mode exp_pos_enc
    python train.py --mode exp_label_smoothing

Translate a sentence (after training):
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset      import get_dataset, PAD_IDX, BOS_IDX, EOS_IDX
from model        import Transformer
from lr_scheduler import get_optimizer_and_scheduler, NoamScheduler


# ══════════════════════════════════════════════════════════════════════════════
# Default hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = dict(
    # Model architecture (scaled down from paper for Multi30k)
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
    save_path      = "checkpoints/best_model.pt",
    wandb_project  = "da6401_assignment3",
    wandb_entity   = None,   # set to your W&B username / team if needed
)


# ══════════════════════════════════════════════════════════════════════════════
# Label-Smoothing Loss
# ══════════════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing (ε_ls = 0.1 as required).

    Instead of a hard one-hot target, distribute ε_ls uniformly over all
    non-pad tokens and assign (1 - ε_ls) to the correct token.

    This acts as a regulariser — the model can no longer become arbitrarily
    confident, which improves generalisation even if it raises perplexity.

    Args:
        vocab_size : target vocabulary size
        pad_idx    : <pad> token index — excluded from loss
        smoothing  : ε_ls (default 0.1)
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
            logits  : (N, vocab_size)  raw model output (flattened)
            targets : (N,)             gold token ids   (flattened)

        Returns:
            loss : scalar
        """
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            # Uniform distribution then override the correct-token slot
            smooth_dist = torch.full_like(log_probs,
                                          self.smoothing / (self.vocab_size - 2))
            smooth_dist[:, self.pad_idx] = 0.0
            smooth_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

            # Zero out padding positions completely
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
    Corpus-level BLEU via the `evaluate` library (sacrebleu backend).

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
                src = src_batch[i].unsqueeze(0)

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
    else:
        # Fallback: simple n-gram BLEU
        from collections import Counter
        def ngrams(tokens, n):
            return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
        total_match, total_pred = 0, 0
        for pred, refs in zip(predictions, references):
            p_tok = pred.split(); r_tok = refs[0].split()
            p4 = ngrams(p_tok, 4); r4 = ngrams(r_tok, 4)
            for ng, cnt in p4.items():
                total_match += min(cnt, r4.get(ng, 0))
            total_pred += max(len(p_tok) - 3, 0)
        precision = total_match / max(total_pred, 1)
        return precision * 100


# ══════════════════════════════════════════════════════════════════════════════
# Core training / evaluation functions
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler, criterion, device,
                clip_grad: float = 1.0):
    """One full training pass.  Returns average loss per non-pad token."""
    model.train()
    total_loss, n_batches = 0.0, 0

    for src, tgt in tqdm(loader, desc="Train", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher forcing: feed tgt[:-1], predict tgt[1:]
        tgt_input  = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)
        B, T, V = logits.size()

        loss = criterion(logits.contiguous().view(-1, V),
                         tgt_output.contiguous().view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        # Noam: update LR BEFORE optimizer.step()
        scheduler.step()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def evaluate_loss(model, loader, criterion, device):
    """Validation loss (no gradients)."""
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

    wandb.init(project=config["wandb_project"], entity=config.get("wandb_entity"),
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
    print(f"Parameters: {n_params:,}")

    criterion             = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX,
                                               config["label_smooth"])
    optimizer, scheduler  = get_optimizer_and_scheduler(
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

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
              f"ppl={train_ppl:.1f} | val_loss={val_loss:.4f} ppl={val_ppl:.1f} "
              f"| lr={scheduler.current_lr:.6f}")

        log = {"epoch": epoch, "train/loss": train_loss,
               "train/perplexity": train_ppl, "val/loss": val_loss,
               "val/perplexity": val_ppl, "lr": scheduler.current_lr,
               "step": scheduler.current_step}

        # BLEU every 5 epochs (slow)
        if epoch % 5 == 0 or epoch == config["num_epochs"]:
            val_bleu = compute_bleu(model, val_loader, tgt_vocab, device)
            log["val/bleu"] = val_bleu
            print(f"          val_bleu={val_bleu:.2f}")

        wandb.log(log)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "src_vocab": src_vocab, "tgt_vocab": tgt_vocab,
                        "config": config}, config["save_path"])
            print(f"          ✓ checkpoint saved (val_loss={val_loss:.4f})")

    # Final test BLEU
    print("\nEvaluating on test set …")
    ckpt = torch.load(config["save_path"], map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_bleu = compute_bleu(model, test_loader, tgt_vocab, device)
    print(f"Test BLEU: {test_bleu:.2f}")
    wandb.log({"test/bleu": test_bleu})
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Inference (translate a single sentence)
# ══════════════════════════════════════════════════════════════════════════════

def translate(checkpoint_path: str, sentence: str, max_len: int = 100):
    """Load checkpoint and translate one German sentence to English."""
    from dataset import load_spacy_models, tokenize_de

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt      = torch.load(checkpoint_path, map_location=device)
    cfg       = ckpt["config"]
    src_vocab = ckpt["src_vocab"]
    tgt_vocab = ckpt["tgt_vocab"]

    model = Transformer(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    spacy_de, _ = load_spacy_models()
    tokens   = tokenize_de(sentence, spacy_de)
    ids      = [BOS_IDX] + src_vocab.encode(tokens) + [EOS_IDX]
    src      = torch.tensor([ids], dtype=torch.long, device=device)

    pred_ids = model.infer(src, BOS_IDX, EOS_IDX, PAD_IDX, max_len)
    pred_ids = [t for t in pred_ids if t not in (EOS_IDX, PAD_IDX)]
    translation = " ".join(tgt_vocab.decode(pred_ids))

    print(f"DE: {sentence}")
    print(f"EN: {translation}")
    return translation


# ══════════════════════════════════════════════════════════════════════════════
# Experiment helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_model_and_data(cfg):
    """Shared setup for all experiments."""
    train_loader, val_loader, _, src_vocab, tgt_vocab = get_dataset(
        batch_size=cfg["batch_size"], min_freq=cfg["min_freq"],
        max_len=cfg["max_len"],
    )
    model = Transformer(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], dropout=cfg["dropout"],
        max_len=cfg["max_len"],
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model, train_loader, val_loader, src_vocab, tgt_vocab


def _simple_train(run_name, model, train_loader, val_loader, cfg,
                  optimizer, scheduler, criterion, device, epochs=10):
    """Minimal training loop for experiments."""
    for epoch in range(1, epochs + 1):
        tl = train_epoch(model, train_loader, optimizer, scheduler,
                         criterion, device, cfg["clip_grad"])
        vl = evaluate_loss(model, val_loader, criterion, device)
        lr_now = scheduler.current_lr if hasattr(scheduler, "current_lr") \
                 else optimizer.param_groups[0]["lr"]
        wandb.log({"epoch": epoch, "train/loss": tl, "val/loss": vl,
                   "lr": lr_now})
        print(f"  [{run_name}] epoch {epoch}: train={tl:.3f} val={vl:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2.1 – Noam Scheduler vs Fixed LR
# ══════════════════════════════════════════════════════════════════════════════

def exp_noam_vs_fixed(cfg, epochs=15):
    """
    Train two identical models:
      (a) Noam scheduler — linear warm-up + inverse-sqrt decay
      (b) Fixed LR = 1e-4

    Purpose: show that the Transformer is sensitive to the initial LR and
    that warm-up prevents early divergence in the self-attention layers.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for use_noam in [True, False]:
        run_name = "noam_scheduler" if use_noam else "fixed_lr_1e4"
        model, train_loader, val_loader, src_vocab, tgt_vocab = \
            _build_model_and_data(cfg)
        criterion = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, cfg["label_smooth"])

        if use_noam:
            opt, sched = get_optimizer_and_scheduler(
                model, cfg["d_model"], cfg["warmup_steps"]
            )
        else:
            opt   = torch.optim.Adam(model.parameters(), lr=1e-4,
                                     betas=(0.9, 0.98), eps=1e-9)
            class FixedSched:
                current_lr   = 1e-4
                current_step = 0
                def step(self): self.current_step += 1
            sched = FixedSched()

        wandb.init(project=cfg["wandb_project"], name=run_name, reinit=True)
        _simple_train(run_name, model, train_loader, val_loader, cfg,
                      opt, sched, criterion, device, epochs)
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2.2 – Scaling Factor 1/√dk Ablation
# ══════════════════════════════════════════════════════════════════════════════

def exp_scaling_factor(cfg, n_steps=1000):
    """
    Compare gradient norms of W_Q / W_K during the first 1000 steps,
    with and without the 1/√d_k scaling factor.

    Without scaling, large dot-products saturate softmax → near-zero gradients
    through the attention layer (the "vanishing gradient" phenomenon discussed
    in paper section 3.2.1).
    """
    import torch.nn.functional as F_

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, train_loader, val_loader, src_vocab, tgt_vocab = _build_model_and_data(cfg)
    criterion = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, cfg["label_smooth"])

    for use_scale in [True, False]:
        run_name = "with_scaling" if use_scale else "without_scaling"
        model, *_ = _build_model_and_data(cfg)

        # Monkey-patch attention if ablating the scale
        if not use_scale:
            import types
            from model import MultiHeadAttention

            def _forward_no_scale(self, query, key, value, mask=None):
                Q = self._split_heads(self.W_Q(query))
                K = self._split_heads(self.W_K(key))
                V = self._split_heads(self.W_V(value))
                scores = torch.matmul(Q, K.transpose(-2, -1))   # NO /√d_k
                if mask is not None:
                    if mask.dim() == 3: mask = mask.unsqueeze(1)
                    scores = scores.masked_fill(mask, -1e9)
                w = F_.softmax(scores, dim=-1)
                out = self._combine_heads(torch.matmul(w, V))
                return self.W_O(self.dropout(out)), w

            for m in model.modules():
                if isinstance(m, MultiHeadAttention):
                    m.forward = types.MethodType(_forward_no_scale, m)

        wandb.init(project=cfg["wandb_project"], name=run_name, reinit=True)
        opt, sched = get_optimizer_and_scheduler(model, cfg["d_model"],
                                                  cfg["warmup_steps"])
        model.train()
        data_iter = iter(train_loader)

        for step in range(1, n_steps + 1):
            try:
                src, tgt = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                src, tgt = next(data_iter)

            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            B, T, V = logits.size()
            loss = criterion(logits.contiguous().view(-1, V),
                             tgt[:, 1:].contiguous().view(-1))
            opt.zero_grad(); loss.backward()

            enc_attn = model.encoder.layers[0].self_attn
            q_norm = enc_attn.W_Q.weight.grad.norm().item() \
                     if enc_attn.W_Q.weight.grad is not None else 0.0
            k_norm = enc_attn.W_K.weight.grad.norm().item() \
                     if enc_attn.W_K.weight.grad is not None else 0.0

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
            sched.step(); opt.step()

            if step % 50 == 0:
                wandb.log({"step": step, "loss": loss.item(),
                           "grad_norm/W_Q": q_norm, "grad_norm/W_K": k_norm})
                print(f"  [{run_name}] step {step} loss={loss.item():.3f} "
                      f"Q_grad={q_norm:.5f} K_grad={k_norm:.5f}")

        wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2.3 – Attention Rollout & Head Specialisation
# ══════════════════════════════════════════════════════════════════════════════

def exp_attn_rollout(cfg, checkpoint_path=None):
    """
    Extract per-head attention from the last encoder layer and log heatmaps.
    """
    from dataset import load_spacy_models, tokenize_de

    if checkpoint_path is None:
        checkpoint_path = cfg["save_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt      = torch.load(checkpoint_path, map_location=device)
    src_vocab = ckpt["src_vocab"]
    tgt_vocab = ckpt["tgt_vocab"]
    c         = ckpt["config"]

    model = Transformer(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
        d_model=c["d_model"], num_layers=c["num_layers"],
        num_heads=c["num_heads"], d_ff=c["d_ff"], dropout=c["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Hook to capture attention weights from last encoder layer
    attn_store = {}
    def _hook(module, inp, out):
        if isinstance(out, tuple):
            attn_store["weights"] = out[1].detach().cpu()
    model.encoder.layers[-1].self_attn.register_forward_hook(_hook)

    spacy_de, _ = load_spacy_models()
    sentence  = "Ein Hund läuft über die Wiese."
    tokens    = tokenize_de(sentence, spacy_de)
    ids       = [BOS_IDX] + src_vocab.encode(tokens) + [EOS_IDX]
    src       = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        model.encoder(src, model.make_src_mask(src))

    attn     = attn_store["weights"][0]   # (heads, seq, seq)
    tok_disp = ["<bos>"] + tokens + ["<eos>"]

    wandb.init(project=cfg["wandb_project"], name="attn_rollout", reinit=True)
    for h in range(attn.size(0)):
        a = attn[h].numpy()
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(a, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(tok_disp)))
        ax.set_xticklabels(tok_disp, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(tok_disp)))
        ax.set_yticklabels(tok_disp, fontsize=8)
        ax.set_title(f"Encoder Last Layer – Head {h+1}")
        plt.colorbar(im, ax=ax); plt.tight_layout()
        wandb.log({f"attention/head_{h+1}": wandb.Image(fig)})
        plt.close(fig)
        print(f"  Logged head {h+1}")
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2.4 – Sinusoidal PE vs Learned Positional Embeddings
# ══════════════════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    """Drop-in replacement: nn.Embedding instead of sinusoidal PE."""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.embedding(pos))


def exp_pos_enc(cfg, epochs=10):
    """
    Compare sinusoidal vs learned positional encoding on validation BLEU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, train_loader, val_loader, src_vocab, tgt_vocab = _build_model_and_data(cfg)
    criterion = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, cfg["label_smooth"])

    for learned in [False, True]:
        run_name = "learned_PE" if learned else "sinusoidal_PE"
        model, *_ = _build_model_and_data(cfg)

        if learned:
            for enc_dec in [model.encoder, model.decoder]:
                enc_dec.pos_enc = LearnedPositionalEncoding(
                    cfg["d_model"], cfg["max_len"], cfg["dropout"]
                ).to(device)

        wandb.init(project=cfg["wandb_project"], name=run_name, reinit=True)
        opt, sched = get_optimizer_and_scheduler(model, cfg["d_model"],
                                                  cfg["warmup_steps"])

        for epoch in range(1, epochs + 1):
            tl = train_epoch(model, train_loader, opt, sched,
                             criterion, device, cfg["clip_grad"])
            vl = evaluate_loss(model, val_loader, criterion, device)
            log = {"epoch": epoch, "train/loss": tl, "val/loss": vl}
            if epoch % 5 == 0:
                vb = compute_bleu(model, val_loader, tgt_vocab, device)
                log["val/bleu"] = vb
                print(f"  [{run_name}] epoch {epoch}: bleu={vb:.2f}")
            wandb.log(log)
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2.5 – Label Smoothing Ablation (ε=0.1 vs ε=0.0)
# ══════════════════════════════════════════════════════════════════════════════

def exp_label_smoothing(cfg, epochs=10):
    """
    Compare ε_ls=0.1 vs ε_ls=0.0 (standard cross-entropy).
    Log the softmax probability of the CORRECT token ("prediction confidence")
    at each epoch to show that label smoothing prevents over-confidence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, train_loader, val_loader, src_vocab, tgt_vocab = _build_model_and_data(cfg)

    for smooth in [0.1, 0.0]:
        run_name  = f"label_smooth_{smooth}"
        model, *_ = _build_model_and_data(cfg)
        criterion  = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smooth)

        wandb.init(project=cfg["wandb_project"], name=run_name, reinit=True)
        opt, sched = get_optimizer_and_scheduler(model, cfg["d_model"],
                                                  cfg["warmup_steps"])

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, total_conf, n_tok, n_batch = 0.0, 0.0, 0, 0

            for src, tgt in train_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in = tgt[:, :-1]; tgt_out = tgt[:, 1:]
                logits = model(src, tgt_in)
                B, T, V = logits.size()
                flat_l = logits.contiguous().view(-1, V)
                flat_t = tgt_out.contiguous().view(-1)

                loss = criterion(flat_l, flat_t)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
                sched.step(); opt.step()

                total_loss += loss.item(); n_batch += 1

                with torch.no_grad():
                    probs  = torch.softmax(flat_l, dim=-1)
                    mask   = flat_t != PAD_IDX
                    conf   = probs[mask].gather(1, flat_t[mask].unsqueeze(1))
                    total_conf += conf.sum().item()
                    n_tok      += mask.sum().item()

            avg_loss = total_loss / max(n_batch, 1)
            avg_conf = total_conf / max(n_tok, 1)
            vl       = evaluate_loss(model, val_loader, criterion, device)

            wandb.log({"epoch": epoch, "train/loss": avg_loss,
                       "val/loss": vl, "train/pred_confidence": avg_conf})
            print(f"  [{run_name}] epoch {epoch}: loss={avg_loss:.3f} "
                  f"conf={avg_conf:.3f} val={vl:.3f}")

        wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train",
                        choices=["train", "translate",
                                 "exp_noam_vs_fixed", "exp_scaling_factor",
                                 "exp_attn_rollout", "exp_pos_enc",
                                 "exp_label_smoothing"])

    # Hyperparameter overrides
    for k, v in DEFAULT_CONFIG.items():
        t = type(v) if v is not None else str
        parser.add_argument(f"--{k}", type=t, default=v)

    # Inference-only args
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

    elif args.mode == "exp_noam_vs_fixed":
        exp_noam_vs_fixed(config)

    elif args.mode == "exp_scaling_factor":
        exp_scaling_factor(config)

    elif args.mode == "exp_attn_rollout":
        exp_attn_rollout(config, checkpoint_path=args.checkpoint)

    elif args.mode == "exp_pos_enc":
        exp_pos_enc(config)

    elif args.mode == "exp_label_smoothing":
        exp_label_smoothing(config)
