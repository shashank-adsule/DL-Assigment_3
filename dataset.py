"""
dataset.py
----------
Tokenisation (spaCy), vocabulary, and DataLoader for Multi30k DE→EN.

Special token indices (imported by every other module):
    PAD_IDX = 0
    BOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import spacy
from collections import Counter

# ── Special tokens ─────────────────────────────────────────────────────────────
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


# ══════════════════════════════════════════════════════════════════════════════
# Vocabulary
# ══════════════════════════════════════════════════════════════════════════════

class Vocabulary:
    """
    Maps tokens ↔ integer indices.

    Fixed layout:
        0  <pad>
        1  <bos>
        2  <eos>
        3  <unk>
        4+ normal tokens (sorted by frequency, then alphabetically)
    """

    SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self):
        self.token2idx = {tok: i for i, tok in enumerate(self.SPECIALS)}
        self.idx2token = {i: tok for i, tok in enumerate(self.SPECIALS)}

    def build(self, tokenized_sentences, min_freq: int = 2):
        """Add tokens that appear ≥ min_freq times across all sentences."""
        counter = Counter()
        for tokens in tokenized_sentences:
            counter.update(tokens)

        for token, freq in sorted(counter.items()):   # deterministic order
            if freq >= min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx]   = token

    def encode(self, tokens):
        """List[str] → List[int]  (unknown → UNK_IDX)."""
        return [self.token2idx.get(t, UNK_IDX) for t in tokens]

    def decode(self, indices):
        """List[int] → List[str]."""
        return [self.idx2token.get(i, "<unk>") for i in indices]

    def __len__(self):
        return len(self.token2idx)


# ══════════════════════════════════════════════════════════════════════════════
# spaCy tokenisers
# ══════════════════════════════════════════════════════════════════════════════

def load_spacy_models():
    """
    Load German and English spaCy tokenisers.

    Install with:
        python -m spacy download de_core_news_sm
        python -m spacy download en_core_web_sm
    """
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except OSError:
        raise OSError("Run: python -m spacy download de_core_news_sm")
    try:
        spacy_en = spacy.load("en_core_web_sm")
    except OSError:
        raise OSError("Run: python -m spacy download en_core_web_sm")
    return spacy_de, spacy_en


def tokenize_de(text, spacy_de):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]


def tokenize_en(text, spacy_en):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):
    """
    Wraps one split of Multi30k into a PyTorch Dataset.

    Each item is (src_tensor, tgt_tensor) where both are 1-D LongTensors
    starting with BOS_IDX and ending with EOS_IDX.
    """

    def __init__(self, hf_split, src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                 spacy_de, spacy_en, max_len: int = 150):
        self.samples = []
        for example in hf_split:
            src_tok = tokenize_de(example["de"], spacy_de)
            tgt_tok = tokenize_en(example["en"], spacy_en)

            if len(src_tok) > max_len or len(tgt_tok) > max_len:
                continue   # skip outliers to avoid memory spikes

            src_ids = [BOS_IDX] + src_vocab.encode(src_tok) + [EOS_IDX]
            tgt_ids = [BOS_IDX] + tgt_vocab.encode(tgt_tok) + [EOS_IDX]

            self.samples.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Collation
# ══════════════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    """Pad a batch of (src, tgt) pairs to the same length."""
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_batch, tgt_batch


# ══════════════════════════════════════════════════════════════════════════════
# Top-level builder  (called from train.py)
# ══════════════════════════════════════════════════════════════════════════════

def get_dataset(batch_size: int = 128, min_freq: int = 2, max_len: int = 150):
    """
    Load Multi30k from HuggingFace JSONL files, build vocabularies,
    return DataLoaders and vocab objects.

    Vocabularies are built on TRAINING data only (no data leakage).

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    import pandas as pd

    print("Loading Multi30k …")
    splits = {
        "train"     : "train.jsonl",
        "validation": "val.jsonl",
        "test"      : "test.jsonl",
    }

    # Load each split as a list of dicts with keys "de" and "en"
    train_raw = pd.read_json(
        "hf://datasets/bentrevett/multi30k/" + splits["train"],
        lines=True
    ).to_dict("records")
    val_raw   = pd.read_json(
        "hf://datasets/bentrevett/multi30k/" + splits["validation"],
        lines=True
    ).to_dict("records")
    test_raw  = pd.read_json(
        "hf://datasets/bentrevett/multi30k/" + splits["test"],
        lines=True
    ).to_dict("records")

    print("Loading spaCy models …")
    spacy_de, spacy_en = load_spacy_models()

    # Build vocabularies from training split only
    print("Building vocabularies …")
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()

    src_tok_list = [tokenize_de(ex["de"], spacy_de) for ex in train_raw]
    tgt_tok_list = [tokenize_en(ex["en"], spacy_en) for ex in train_raw]

    src_vocab.build(src_tok_list, min_freq=min_freq)
    tgt_vocab.build(tgt_tok_list, min_freq=min_freq)

    print(f"  |src_vocab| = {len(src_vocab)}")
    print(f"  |tgt_vocab| = {len(tgt_vocab)}")

    # Build datasets
    train_ds = Multi30kDataset(train_raw, src_vocab, tgt_vocab, spacy_de, spacy_en, max_len)
    val_ds   = Multi30kDataset(val_raw,   src_vocab, tgt_vocab, spacy_de, spacy_en, max_len)
    test_ds  = Multi30kDataset(test_raw,  src_vocab, tgt_vocab, spacy_de, spacy_en, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab