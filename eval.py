"""
eval.py
-------
Standalone local BLEU evaluator for the DE→EN Transformer.

Loads your checkpoint, runs beam-search decoding on val + test splits,
prints a detailed diagnostic report, and optionally saves all predictions
to a TSV file so you can inspect individual translations.

Usage
-----
# Basic — evaluate best checkpoint on val + test:
    python eval.py

# Specify a checkpoint explicitly:
    python eval.py --checkpoint checkpoints/best_model.pt

# Fast sanity check — only first 200 sentences, greedy decoding:
    python eval.py --max_samples 200 --beam_size 1

# Compare greedy vs beam search:
    python eval.py --compare_beams

# Translate a single sentence and exit:
    python eval.py --sentence "Ein Hund spielt im Park."

# Save all predictions to a TSV for manual inspection:
    python eval.py --save_predictions predictions.tsv

# Quiet — just print the final BLEU number:
    python eval.py --quiet
"""

import os
import sys
import math
import time
import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Import project modules ────────────────────────────────────────────────────
try:
    from dataset import (get_dataset, PAD_IDX, BOS_IDX, EOS_IDX,
                         load_spacy_models, tokenize_de, Vocabulary)
    from model   import Transformer, load_checkpoint
except ImportError as e:
    sys.exit(f"[eval.py] Import error: {e}\n"
             f"Make sure eval.py is in the same directory as model.py and dataset.py")


# ══════════════════════════════════════════════════════════════════════════════
# BLEU helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def corpus_bleu_manual(predictions, references):
    """
    Full corpus BLEU with brevity penalty and per-n-gram precisions.
    Returns (bleu_score, [p1, p2, p3, p4], bp).
    """
    clip_matches   = [0] * 4
    total_pred_n   = [0] * 4
    total_pred_len = 0
    total_ref_len  = 0

    for pred, ref in zip(predictions, references):
        p_toks = pred.split()
        r_toks = ref.split()
        total_pred_len += len(p_toks)
        total_ref_len  += len(r_toks)
        for n in range(1, 5):
            p_ng = _ngrams(p_toks, n)
            r_ng = _ngrams(r_toks, n)
            for ng, cnt in p_ng.items():
                clip_matches[n-1] += min(cnt, r_ng.get(ng, 0))
            total_pred_n[n-1] += max(len(p_toks) - n + 1, 0)

    precisions = []
    for m, t in zip(clip_matches, total_pred_n):
        precisions.append(m / t if t > 0 else 0.0)

    if any(p == 0 for p in precisions):
        return 0.0, precisions, 1.0

    log_avg = sum(math.log(p) for p in precisions) / 4
    bp      = min(1.0, math.exp(1 - total_ref_len / max(total_pred_len, 1)))
    score   = bp * math.exp(log_avg) * 100
    return score, precisions, bp


def compute_bleu(score, precisions, bp):
    """Pretty-print a BLEU breakdown."""
    p1, p2, p3, p4 = [p * 100 for p in precisions]
    return (f"BLEU = {score:.2f}  "
            f"(1-gram: {p1:.1f} / 2-gram: {p2:.1f} / "
            f"3-gram: {p3:.1f} / 4-gram: {p4:.1f})  BP={bp:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Core evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, loader, tgt_vocab, device,
             beam_size=4, max_len=100, max_samples=None,
             quiet=False, split_name="val"):
    """
    Run beam-search over a DataLoader and return predictions + references.

    Returns:
        predictions : List[str]   one translated sentence per sample
        references  : List[str]   one reference sentence per sample
        elapsed_sec : float
    """
    model.eval()
    predictions = []
    references  = []
    n_done      = 0
    t0          = time.time()

    iterator = tqdm(loader, desc=f"  Decoding {split_name}", disable=quiet)

    with torch.no_grad():
        for src_batch, tgt_batch in iterator:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            for i in range(src_batch.size(0)):
                if max_samples and n_done >= max_samples:
                    break

                src = src_batch[i].unsqueeze(0)

                pred_ids = model.infer(
                    src, BOS_IDX, EOS_IDX, PAD_IDX,
                    max_len=max_len,
                    beam_size=beam_size,
                    return_tokens=True,
                )

                ref_ids = [t for t in tgt_batch[i].tolist()
                           if t not in (BOS_IDX, EOS_IDX, PAD_IDX)]

                predictions.append(" ".join(tgt_vocab.decode(pred_ids)))
                references.append(" ".join(tgt_vocab.decode(ref_ids)))
                n_done += 1

            if max_samples and n_done >= max_samples:
                break

    elapsed = time.time() - t0
    return predictions, references, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic helpers
# ══════════════════════════════════════════════════════════════════════════════

def length_bucket_bleu(predictions, references, buckets=None):
    """
    Break BLEU down by reference sentence length bucket.
    Helps diagnose whether the model struggles on short or long sentences.
    """
    if buckets is None:
        buckets = [(1, 10), (11, 20), (21, 30), (31, 50), (51, 9999)]

    results = []
    for lo, hi in buckets:
        preds = [p for p, r in zip(predictions, references)
                 if lo <= len(r.split()) <= hi]
        refs  = [r for p, r in zip(predictions, references)
                 if lo <= len(r.split()) <= hi]
        if not preds:
            continue
        score, prec, bp = corpus_bleu_manual(preds, refs)
        label = f"{lo:>3}–{hi if hi < 9999 else '∞':>3} tokens"
        results.append((label, score, len(preds)))
    return results


def worst_translations(predictions, references, n=10):
    """
    Find sentences where the model did worst (by sentence-level 1-gram precision).
    Good for qualitative diagnosis.
    """
    scored = []
    for pred, ref in zip(predictions, references):
        p_toks = pred.split()
        r_toks = set(ref.split())
        if not p_toks:
            scored.append((0.0, pred, ref))
            continue
        hits = sum(1 for t in p_toks if t in r_toks)
        scored.append((hits / len(p_toks), pred, ref))

    scored.sort(key=lambda x: x[0])
    return scored[:n]


def best_translations(predictions, references, n=5):
    """Find sentences where the model did best."""
    scored = []
    for pred, ref in zip(predictions, references):
        p_toks = pred.split()
        r_toks = set(ref.split())
        if not p_toks:
            continue
        hits = sum(1 for t in p_toks if t in r_toks)
        scored.append((hits / len(p_toks), pred, ref))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:n]


def repetition_rate(predictions):
    """Fraction of predictions that contain repeated trigrams (degenerate output)."""
    flagged = 0
    for pred in predictions:
        toks = pred.split()
        if len(toks) < 3:
            continue
        tgs = [tuple(toks[i:i+3]) for i in range(len(toks)-2)]
        if len(tgs) != len(set(tgs)):
            flagged += 1
    return flagged / max(len(predictions), 1)


def avg_length_ratio(predictions, references):
    pred_lens = [len(p.split()) for p in predictions]
    ref_lens  = [len(r.split()) for r in references]
    if not ref_lens:
        return 0.0, 0.0, 0.0
    return (sum(pred_lens) / len(pred_lens),
            sum(ref_lens)  / len(ref_lens),
            sum(pred_lens) / max(sum(ref_lens), 1))


# ══════════════════════════════════════════════════════════════════════════════
# Report printer
# ══════════════════════════════════════════════════════════════════════════════

SEP  = "=" * 70
SEP2 = "-" * 70

def print_report(split_name, predictions, references, elapsed,
                 beam_size, max_samples, quiet):
    score, prec, bp = corpus_bleu_manual(predictions, references)

    # Try sacrebleu for a second opinion
    sacre_score = None
    try:
        import sacrebleu as sb
        sacre_score = sb.corpus_bleu(predictions, [references]).score
    except ImportError:
        pass

    avg_pred, avg_ref, ratio = avg_length_ratio(predictions, references)
    rep = repetition_rate(predictions)
    n   = len(predictions)

    print(f"\n{SEP}")
    print(f"  {split_name.upper()} SET RESULTS"
          + (f"  (first {max_samples} samples)" if max_samples else ""))
    print(SEP)
    print(f"  Sentences decoded : {n}")
    print(f"  Beam size         : {beam_size}")
    print(f"  Elapsed           : {elapsed:.1f}s  ({elapsed/max(n,1)*1000:.0f}ms/sent)")
    print(SEP2)
    print(f"  {compute_bleu(score, prec, bp)}")
    if sacre_score is not None:
        print(f"  sacrebleu (tok13a): {sacre_score:.2f}")
    print(SEP2)
    print(f"  Avg pred length   : {avg_pred:.1f} tokens")
    print(f"  Avg ref length    : {avg_ref:.1f} tokens")
    print(f"  Length ratio      : {ratio:.3f}  (BP = {bp:.3f})")
    print(f"  Repetition rate   : {rep*100:.1f}%  "
          "← should be <5%; higher means model is looping")
    print(SEP2)

    # Length bucket breakdown
    buckets = length_bucket_bleu(predictions, references)
    print("  BLEU by reference length:")
    for label, bscore, cnt in buckets:
        bar = "█" * int(bscore / 2)
        print(f"    {label}  n={cnt:>4}  {bscore:5.2f}  {bar}")

    if not quiet:
        # Worst translations
        print(f"\n{SEP2}")
        print("  WORST 5 TRANSLATIONS  (low 1-gram precision)")
        print(SEP2)
        for rank, (sc, pred, ref) in enumerate(
                worst_translations(predictions, references, n=5), 1):
            print(f"  [{rank}] precision={sc:.2f}")
            print(f"      REF : {ref}")
            print(f"      PRED: {pred}")

        # Best translations
        print(f"\n{SEP2}")
        print("  BEST 3 TRANSLATIONS")
        print(SEP2)
        for rank, (sc, pred, ref) in enumerate(
                best_translations(predictions, references, n=3), 1):
            print(f"  [{rank}] precision={sc:.2f}")
            print(f"      REF : {ref}")
            print(f"      PRED: {pred}")

    print(SEP)
    return score


# ══════════════════════════════════════════════════════════════════════════════
# Beam-size comparison
# ══════════════════════════════════════════════════════════════════════════════

def compare_beams(model, loader, tgt_vocab, device,
                  beam_sizes=(1, 2, 4, 8), max_samples=500):
    """
    Run decoding with several beam sizes on the first max_samples sentences
    and print a comparison table. Useful for picking the right beam_size
    before running the full evaluation.
    """
    print(f"\n{SEP}")
    print(f"  BEAM SIZE COMPARISON  (first {max_samples} samples)")
    print(SEP)
    print(f"  {'Beam':>5}  {'BLEU':>7}  {'Time(s)':>8}  {'ms/sent':>8}")
    print(SEP2)

    for bs in beam_sizes:
        preds, refs, elapsed = evaluate(
            model, loader, tgt_vocab, device,
            beam_size=bs, max_samples=max_samples, quiet=True,
        )
        score, _, _ = corpus_bleu_manual(preds, refs)
        mps = elapsed / max(len(preds), 1) * 1000
        print(f"  {bs:>5}  {score:>7.2f}  {elapsed:>8.1f}  {mps:>8.0f}")

    print(SEP)


# ══════════════════════════════════════════════════════════════════════════════
# Save predictions TSV
# ══════════════════════════════════════════════════════════════════════════════

def save_predictions_tsv(path, predictions, references, split_name):
    """
    Save a TSV with columns: split | reference | prediction
    Open in Excel / any spreadsheet tool for easy manual review.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("split\treference\tprediction\n")
        for ref, pred in zip(references, predictions):
            f.write(f"{split_name}\t{ref}\t{pred}\n")
    print(f"  Predictions saved to: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Single-sentence translation
# ══════════════════════════════════════════════════════════════════════════════

def translate_sentence(model, sentence, beam_size=4):
    """Translate one sentence and show it."""
    print(f"\n{'─'*60}")
    print(f"  DE : {sentence}")
    t0 = time.time()
    translation = model.infer(sentence, beam_size=beam_size)
    elapsed = (time.time() - t0) * 1000
    print(f"  EN : {translation}")
    print(f"  ({elapsed:.0f}ms, beam={beam_size})")
    print(f"{'─'*60}")
    return translation


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Local BLEU evaluator for DE→EN Transformer",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best_model.pt",
        help="Path to .pt checkpoint  (default: checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--beam_size", type=int, default=4,
        help="Beam width for decoding  (default: 4)",
    )
    parser.add_argument(
        "--max_len", type=int, default=100,
        help="Max tokens to generate per sentence  (default: 100)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="DataLoader batch size  (default: 64)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit evaluation to first N sentences (default: all)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["val", "test"],
        choices=["val", "test", "train"],
        help="Which splits to evaluate  (default: val test)",
    )
    parser.add_argument(
        "--compare_beams", action="store_true",
        help="Compare BLEU vs beam size (1,2,4,8) before main eval",
    )
    parser.add_argument(
        "--sentence", default=None,
        help="Translate a single German sentence and exit",
    )
    parser.add_argument(
        "--save_predictions", default=None, metavar="PATH",
        help="Save predictions + references to a TSV file",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-sentence examples; just print the BLEU numbers",
    )
    parser.add_argument(
        "--device", default=None,
        help="Force device: 'cpu' or 'cuda'  (default: auto-detect)",
    )
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{SEP}")
    print("  eval.py — DE→EN Transformer  local BLEU evaluator")
    print(SEP)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Beam size  : {args.beam_size}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        sys.exit(f"\n[eval.py] Checkpoint not found: {args.checkpoint}\n"
                 f"  Run `python train.py` first, or check your path.")

    print(f"\nLoading checkpoint …")
    t0 = time.time()
    model = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    print(f"  ✓ Loaded in {time.time()-t0:.1f}s")

    cfg = None
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg  = ckpt.get("config", {})
        print(f"  Epoch saved : {ckpt.get('epoch', '?')}")
        print(f"  Config      : d_model={cfg.get('d_model','?')}  "
              f"layers={cfg.get('num_layers','?')}  "
              f"heads={cfg.get('num_heads','?')}  "
              f"d_ff={cfg.get('d_ff','?')}")
        print(f"  src_vocab   : {len(model.src_vocab)}")
        print(f"  tgt_vocab   : {len(model.tgt_vocab)}")
    except Exception as e:
        print(f"  Warning: could not read config — {e}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters  : {n_params:,}")

    # ── Single-sentence mode ──────────────────────────────────────────────────
    if args.sentence:
        translate_sentence(model, args.sentence, beam_size=args.beam_size)
        sys.exit(0)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading dataset …")
    min_freq = cfg.get("min_freq", 2) if cfg else 2
    max_len  = cfg.get("max_len", 150) if cfg else 150

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataset(
        batch_size=args.batch_size,
        min_freq=min_freq,
        max_len=max_len,
    )

    split_loaders = {
        "train": train_loader,
        "val"  : val_loader,
        "test" : test_loader,
    }

    # Attach dataset vocab to model (may differ from checkpoint vocab if
    # dataset was rebuilt — use checkpoint vocab which is the ground truth)
    tgt_vocab = model.tgt_vocab   # always use checkpoint vocab for decoding

    # ── Beam comparison ───────────────────────────────────────────────────────
    if args.compare_beams:
        compare_beams(model, val_loader, tgt_vocab, device,
                      beam_sizes=[1, 2, 4, 8],
                      max_samples=args.max_samples or 500)

    # ── Main evaluation ───────────────────────────────────────────────────────
    all_results = {}

    for split in args.splits:
        loader = split_loaders[split]

        print(f"\nDecoding {split} split …")
        preds, refs, elapsed = evaluate(
            model, loader, tgt_vocab, device,
            beam_size=args.beam_size,
            max_len=args.max_len,
            max_samples=args.max_samples,
            quiet=args.quiet,
            split_name=split,
        )

        score = print_report(split, preds, refs, elapsed,
                             args.beam_size, args.max_samples, args.quiet)
        all_results[split] = score

        if args.save_predictions:
            # For multiple splits, suffix the filename
            if len(args.splits) > 1:
                base, ext = os.path.splitext(args.save_predictions)
                tsv_path = f"{base}_{split}{ext}"
            else:
                tsv_path = args.save_predictions
            save_predictions_tsv(tsv_path, preds, refs, split)

    # ── Summary table ─────────────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{SEP}")
        print("  SUMMARY")
        print(SEP2)
        for split, score in all_results.items():
            bar = "█" * int(score / 2)
            print(f"  {split:<6}  {score:6.2f}  {bar}")
        print(SEP)

    # ── Actionable advice ─────────────────────────────────────────────────────
    val_score = all_results.get("val", all_results.get("test", 0))
    print("\n  DIAGNOSIS & NEXT STEPS")
    print(SEP2)

    if val_score < 15:
        print("  ⚠  BLEU < 15: Model likely hasn't converged or there is a bug.")
        print("     → Check training loss is actually decreasing each epoch.")
        print("     → Make sure scheduler.step() comes AFTER optimizer.step().")
        print("     → Try --beam_size 1 to rule out a beam search bug.")
    elif val_score < 25:
        print("  ⚠  BLEU 15–25: Model is learning but undertrained or misconfigured.")
        print("     → Train more epochs (50+ recommended for Multi30k).")
        print("     → Confirm d_model=512, num_layers=6, d_ff=2048 (paper base).")
        print("     → Check weight tying is active (output_projection.weight = embedding.weight).")
        print("     → Increase beam size (try --beam_size 8).")
    elif val_score < 32:
        print("  ✓  BLEU 25–32: Good progress. Squeeze out the last points with:")
        print("     → Checkpoint averaging (last 5 epochs).")
        print("     → More epochs — keep training if val BLEU is still rising.")
        print("     → Beam size 6–8 may add 0.5–1 BLEU.")
        print("     → Confirm label smoothing ε=0.1 is active.")
    elif val_score < 37:
        print("  ✓  BLEU 32–37: Strong result. To push further:")
        print("     → Try beam size 8 or 12.")
        print("     → Average more checkpoints (8–10 instead of 5).")
        print("     → Check length-bucket breakdown — fix the weak bucket.")
    else:
        print("  ✓✓ BLEU > 37: Excellent! This matches or exceeds the paper's")
        print("     base model result on Multi30k. You're done.")

    print(SEP)


if __name__ == "__main__":
    main()
