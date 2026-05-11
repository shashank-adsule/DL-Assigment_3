"""
model.py
--------
Full Transformer for sequence-to-sequence translation, implemented from
scratch using only basic PyTorch building-blocks (nn.Linear, nn.Module …).

Sections:
  1.  scaled_dot_product_attention()   – Task 1
  2.  MultiHeadAttention               – Task 1
  3.  PositionalEncoding               – Task 2
  4.  PositionwiseFeedForward          – Task 2
  5.  EncoderLayer + Encoder           – Task 2
  6.  DecoderLayer + Decoder           – Task 2
  7.  Transformer                      – full model + mask helpers + greedy decode
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1. Scaled Dot-Product Attention
# ══════════════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

    Args:
        Q    : (..., seq_q, d_k)
        K    : (..., seq_k, d_k)
        V    : (..., seq_k, d_v)
        mask : bool tensor broadcastable to (..., seq_q, seq_k)
               True  → block that position (fill with -1e9 before softmax)

    Returns:
        output       : (..., seq_q, d_v)
        attn_weights : (..., seq_q, seq_k)   useful for visualisation
    """
    d_k = Q.size(-1)

    # ① Raw attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # ② Apply mask: blocked positions receive -∞ → softmax ≈ 0
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    # ③ Softmax over key dimension
    attn_weights = F.softmax(scores, dim=-1)

    # ④ Weighted sum of values
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


# ══════════════════════════════════════════════════════════════════════════════
# 2. Multi-Head Attention
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (paper section 3.2.2).

    Projects Q/K/V with h parallel linear layers, runs scaled dot-product
    attention for each head, concatenates, and projects back.

        MultiHead(Q,K,V) = Concat(head_1,…,head_h) W^O
        head_i           = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    NOTE: nn.MultiheadAttention is intentionally NOT used (assignment rule).

    Args:
        d_model   : model dimension (must be divisible by num_heads)
        num_heads : number of parallel attention heads
        dropout   : dropout on the output projection
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model {d_model} must be divisible by num_heads {num_heads}"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # dimension per head

        # Four projection matrices (all d_model → d_model)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x):
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x):
        """(batch, heads, seq, d_k) → (batch, seq, d_model)"""
        B, _, S, _ = x.size()
        return x.transpose(1, 2).contiguous().view(B, S, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query : (batch, seq_q, d_model)
            key   : (batch, seq_k, d_model)
            value : (batch, seq_k, d_model)
            mask  : bool tensor broadcastable to (batch, heads, seq_q, seq_k)
                    True → block that (query, key) pair

        Returns:
            output : (batch, seq_q, d_model)
        """
        # Linear projections + reshape into per-head slices
        Q = self._split_heads(self.W_Q(query))   # (B, h, seq_q, d_k)
        K = self._split_heads(self.W_K(key))     # (B, h, seq_k, d_k)
        V = self._split_heads(self.W_V(value))   # (B, h, seq_k, d_k)

        # Expand mask from (B, 1, seq_q, seq_k) → broadcasts over heads
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)   # (B, 1, seq_q, seq_k)

        # Parallel scaled dot-product attention
        attn_out, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_out: (B, h, seq_q, d_k)
        # self.attn_weights stored so EncoderLayer/DecoderLayer can access if needed

        # Concatenate heads and project
        output = self.W_O(self._combine_heads(attn_out))   # (B, seq_q, d_model)
        output = self.dropout(output)

        return output


# ══════════════════════════════════════════════════════════════════════════════
# 3. Positional Encoding
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (paper section 3.5):

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    The table is pre-computed and stored as a **buffer** (not a trainable
    Parameter), so it moves with the model but is never updated by the
    optimiser.  The autograder checks for this specifically.

    Args:
        d_model : embedding / model dimension
        max_len : maximum sequence length to pre-compute (default 5000)
        dropout : applied after adding PE to embeddings
    """

    def __init__(self, d_model: int, max_len: int = 5000,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the PE table: (1, max_len, d_model)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()   # (max_len, 1)

        # Log-space trick for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )   # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)   # even columns
        pe[:, 1::2] = torch.cos(position * div_term)   # odd  columns

        self.register_buffer("pe", pe.unsqueeze(0))    # (1, max_len, d_model)

    def forward(self, x):
        """
        Args:
            x : (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)  with PE added
        """
        x = x + self.pe[:, : x.size(1)]   # broadcast over batch dimension
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Position-wise Feed-Forward Network
# ══════════════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = max(0,  x W_1 + b_1) W_2 + b_2

    Applied identically to every position independently.

    Args:
        d_model : input / output dimension
        d_ff    : inner (hidden) dimension  (paper: 4 × d_model)
        dropout : applied between the two linear layers
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_ff)
        self.fc2     = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# ══════════════════════════════════════════════════════════════════════════════
# 5. Encoder Layer + Encoder Stack
# ══════════════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder layer (paper Figure 1, left):

        Sub-layer 1: Multi-Head Self-Attention  → Add & Norm
        Sub-layer 2: Position-wise FFN          → Add & Norm

    Using Post-LayerNorm  (x = LayerNorm(x + SubLayer(x))) to match the
    paper exactly. Justification: the Noam warm-up scheduler prevents early
    divergence, making Post-LN viable.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x, src_mask=None):
        # Sub-layer 1
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Sub-layer 2
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    """
    Token embedding → positional encoding → N × EncoderLayer.

    Args:
        src_vocab_size : |V_src|
        d_model        : model / embedding dimension
        num_layers     : N  (paper base: 6)
        num_heads      : attention heads per layer
        d_ff           : FFN inner dimension
        dropout        : dropout probability
        max_len        : maximum source sequence length
    """

    def __init__(self, src_vocab_size: int, d_model: int, num_layers: int,
                 num_heads: int, d_ff: int, dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.d_model   = d_model

    def forward(self, src, src_mask=None):
        """
        Args:
            src      : (batch, src_len)  token ids
            src_mask : (batch, 1, 1, src_len) padding mask

        Returns:
            x : (batch, src_len, d_model)
        """
        # Scale embeddings by √d_model (paper section 3.4)
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 6. Decoder Layer + Decoder Stack
# ══════════════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder layer (paper Figure 1, right):

        Sub-layer 1: Masked Multi-Head Self-Attention  → Add & Norm
        Sub-layer 2: Multi-Head Cross-Attention        → Add & Norm
        Sub-layer 3: Position-wise FFN                 → Add & Norm

    The causal mask in sub-layer 1 prevents positions from attending to
    future tokens (autoregressive property).
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        """
        Args:
            x        : (batch, tgt_len, d_model)
            enc_out  : (batch, src_len, d_model)
            src_mask : encoder padding mask
            tgt_mask : combined causal + padding mask

        Returns:
            x            : (batch, tgt_len, d_model)
            self_attn_w  : (batch, heads, tgt_len, tgt_len)
            cross_attn_w : (batch, heads, tgt_len, src_len)
        """
        # Sub-layer 1: masked self-attention
        self_out = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_out))

        # Sub-layer 2: cross-attention (queries from decoder, keys/values from encoder)
        cross_out = self.cross_attn(x, enc_out, enc_out, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_out))

        # Sub-layer 3: FFN
        x = self.norm3(x + self.dropout(self.ffn(x)))

        return x


class Decoder(nn.Module):
    """
    Token embedding → positional encoding → N × DecoderLayer.

    Args: (same as Encoder but for the target language)
    """

    def __init__(self, tgt_vocab_size: int, d_model: int, num_layers: int,
                 num_heads: int, d_ff: int, dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.d_model   = d_model

    def forward(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        """
        Returns:
            x             : (batch, tgt_len, d_model)
            all_cross_attn: list[tensor] one cross-attn map per layer
        """
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 7. Full Transformer
# ══════════════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Encoder-Decoder Transformer for neural machine translation.

    Args:
        src_vocab_size : source vocabulary size
        tgt_vocab_size : target vocabulary size
        d_model        : 512  (paper base)
        num_layers     : 6    (paper base)
        num_heads      : 8
        d_ff           : 2048 (paper base)
        dropout        : 0.1
        max_len        : 5000
    """

    # ── Google Drive file IDs ──────────────────────────────────────────────────
    # After training, upload your .pt files to Google Drive (Anyone with link)
    # and paste the file IDs here.  The autograder will download them
    # automatically when Transformer() is instantiated.
    #
    # Share link format:
    #   https://drive.google.com/file/d/  FILE_ID  /view?usp=sharing
    #
    # ↓↓ PASTE YOUR FILE IDs HERE AFTER TRAINING ↓↓
    GDRIVE_CHECKPOINT_ID = "1tV0glBlcWqXBStgurv3oY4jhkhZyBqrG"
    GDRIVE_VOCAB_ID       = "1C58fGtCeQ4Us9nLXrEBQrv8c-xIsFuGC"
    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    CHECKPOINT_PATH = "checkpoints/best_model.pt"
    VOCAB_PATH      = "checkpoints/vocab.pt"

    @classmethod
    def _download_if_missing(cls):
        """
        Download checkpoint and vocab from Google Drive if not present locally.
        Called automatically in __init__ so the autograder never needs local files.
        """
        import os

        files = [
            (cls.GDRIVE_CHECKPOINT_ID, cls.CHECKPOINT_PATH),
            (cls.GDRIVE_VOCAB_ID,      cls.VOCAB_PATH),
        ]

        for file_id, dest_path in files:
            if file_id.startswith("YOUR_"):
                continue   # not configured yet
            if os.path.exists(dest_path):
                continue   # already downloaded

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            print(f"Downloading {dest_path} from Google Drive …")

            try:
                import gdown
                gdown.download(
                    f"https://drive.google.com/uc?id={file_id}",
                    dest_path, quiet=False
                )
            except ImportError:
                import requests
                url     = f"https://drive.google.com/uc?export=download&id={file_id}"
                session = requests.Session()
                resp    = session.get(url, stream=True)
                # Handle large-file confirmation token
                token   = None
                for k, v in resp.cookies.items():
                    if "download_warning" in k:
                        token = v
                if token:
                    resp = session.get(url + f"&confirm={token}", stream=True)
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(32768):
                        if chunk:
                            f.write(chunk)
            print(f"  ✓ Saved to {dest_path}")

    def __init__(self, src_vocab_size: int = 8000, tgt_vocab_size: int = 8000,
                 d_model: int = 512, num_layers: int = 6,
                 num_heads: int = 8, d_ff: int = 2048,
                 dropout: float = 0.1, max_len: int = 5000):
        # ── Download checkpoints from Google Drive if missing ──────────────────
        self._download_if_missing()

        # ── Override vocab sizes from checkpoint so no size mismatch ──────────
        # The autograder calls Transformer() with no args (default 8000/8000)
        # but the real vocab sizes are stored in the checkpoint.
        import os, torch as _torch
        if os.path.exists(self.CHECKPOINT_PATH):
            try:
                _ckpt = _torch.load(self.CHECKPOINT_PATH,
                                    map_location="cpu", weights_only=False)
                _cfg  = _ckpt.get("config", {})
                src_vocab_size = len(_ckpt["src_vocab"])
                tgt_vocab_size = len(_ckpt["tgt_vocab"])
                # Also override architecture from saved config
                d_model    = _cfg.get("d_model",    d_model)
                num_layers = _cfg.get("num_layers", num_layers)
                num_heads  = _cfg.get("num_heads",  num_heads)
                d_ff       = _cfg.get("d_ff",       d_ff)
                dropout    = _cfg.get("dropout",    dropout)
                max_len    = _cfg.get("max_len",    max_len)
            except Exception as e:
                print(f"  Warning: could not read checkpoint config — {e}")

        super().__init__()

        self.encoder = Encoder(src_vocab_size, d_model, num_layers,
                               num_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers,
                               num_heads, d_ff, dropout, max_len)

        self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.output_projection.weight = self.decoder.embedding.weight

        self._init_weights()

        # ── Load weights ───────────────────────────────────────────────────────
        if os.path.exists(self.CHECKPOINT_PATH):
            try:
                ckpt = _torch.load(self.CHECKPOINT_PATH,
                                   map_location="cpu", weights_only=False)
                self.load_state_dict(ckpt["model_state"], strict=True)
                self.src_vocab = ckpt["src_vocab"]
                self.tgt_vocab = ckpt["tgt_vocab"]
                print("  ✓ Weights and vocab loaded from checkpoint")
            except Exception as e:
                print(f"  Warning: could not load weights — {e}")

        # ── Load vocab (fallback from vocab.pt) ───────────────────────────────
        if not hasattr(self, "src_vocab") and os.path.exists(self.VOCAB_PATH):
            try:
                v = _torch.load(self.VOCAB_PATH,
                                map_location="cpu", weights_only=False)
                self.src_vocab = v["src_vocab"]
                self.tgt_vocab = v["tgt_vocab"]
                print("  ✓ Vocab loaded from vocab.pt")
            except Exception as e:
                print(f"  Warning: could not load vocab — {e}")

    def _init_weights(self):
        """Xavier uniform initialisation (standard for Transformers)."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Mask builders ─────────────────────────────────────────────────────────

    @staticmethod
    def make_src_mask(src, pad_idx: int = 0):
        """
        Encoder padding mask.
        Positions with pad_idx receive True → blocked in attention.

        src : (batch, src_len)
        Returns: (batch, 1, 1, src_len)
        """
        return (src == pad_idx).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_tgt_mask(tgt, pad_idx: int = 0):
        """
        Combined causal + padding mask for the decoder.

        causal: upper-triangular True  → future positions blocked
        pad:    True where tgt == pad_idx

        tgt : (batch, tgt_len)
        Returns: (batch, 1, tgt_len, tgt_len)
        """
        tgt_len = tgt.size(1)
        device  = tgt.device

        # Look-ahead (causal) mask — shape (1, 1, tgt_len, tgt_len)
        causal = torch.triu(
            torch.ones(tgt_len, tgt_len, device=device), diagonal=1
        ).bool().unsqueeze(0).unsqueeze(0)

        # Padding mask — shape (batch, 1, 1, tgt_len)
        pad = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

        return causal | pad   # (batch, 1, tgt_len, tgt_len)

    # ── Forward (teacher-forcing, used during training) ───────────────────────

    def forward(self, src, tgt, pad_idx: int = 0):
        """
        Args:
            src     : (batch, src_len)   source token ids
            tgt     : (batch, tgt_len)   target token ids (teacher-forced input;
                       includes <bos>, excludes final <eos>)
            pad_idx : padding token index

        Returns:
            logits  : (batch, tgt_len, tgt_vocab_size)  raw (un-normalised) scores
        """
        src_mask = self.make_src_mask(src, pad_idx)
        tgt_mask = self.make_tgt_mask(tgt, pad_idx)

        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        logits  = self.output_projection(dec_out)

        return logits

    # ── Greedy decoding (used during inference) ───────────────────────────────

    @torch.no_grad()
    def infer(self, src, bos_idx: int = 1, eos_idx: int = 2,
              pad_idx: int = 0, max_len: int = 50):
        """
        Greedy decoding. Accepts either:
          - a raw German string  → tokenises, decodes, returns English string
          - a LongTensor (1, src_len) → returns English string

        The autograder calls:  model.infer(german_string)

        Args:
            src     : str  OR  LongTensor (1, src_len)
            bos_idx : <bos> index (default 1)
            eos_idx : <eos> index (default 2)
            pad_idx : <pad> index (default 0)
            max_len : maximum tokens to generate

        Returns:
            translation : English string
        """
        # ── Load vocab if not already attached ────────────────────────────────
        # The autograder loads only model weights; vocab must come from the
        # checkpoint file saved alongside the model.
        if not hasattr(self, "src_vocab") or self.src_vocab is None:
            import os, torch as _torch
            vocab_candidates = [
                "checkpoints/vocab.pt",
                "vocab.pt",
                os.path.join(os.path.dirname(
                    os.path.abspath(__file__)), "checkpoints", "vocab.pt"),
            ]
            for path in vocab_candidates:
                if os.path.exists(path):
                    vocab_data = _torch.load(path, weights_only=False)
                    self.src_vocab = vocab_data["src_vocab"]
                    self.tgt_vocab = vocab_data["tgt_vocab"]
                    break
            else:
                # Try downloading vocab.pt from Google Drive as last resort
                try:
                    from train import ensure_checkpoints
                    ensure_checkpoints()
                    if os.path.exists("checkpoints/vocab.pt"):
                        vocab_data = _torch.load("checkpoints/vocab.pt",
                                                 weights_only=False)
                        self.src_vocab = vocab_data["src_vocab"]
                        self.tgt_vocab = vocab_data["tgt_vocab"]
                except Exception:
                    pass

        # ── If src is a raw string, tokenise + encode ─────────────────────────
        if isinstance(src, str):
            try:
                import spacy
                spacy_de = spacy.load("de_core_news_sm")
                tokens   = [tok.text.lower() for tok in spacy_de.tokenizer(src)]
            except Exception:
                tokens = src.lower().split()

            unk = 3
            if hasattr(self, "src_vocab") and self.src_vocab is not None:
                ids = ([bos_idx]
                       + [self.src_vocab.token2idx.get(t, unk) for t in tokens]
                       + [eos_idx])
            else:
                ids = [bos_idx] + [unk] * len(tokens) + [eos_idx]

            device = next(self.parameters()).device
            src    = torch.tensor([ids], dtype=torch.long, device=device)

        # ── Greedy decoding ───────────────────────────────────────────────────
        device   = src.device
        src_mask = self.make_src_mask(src, pad_idx)
        enc_out  = self.encoder(src, src_mask)

        tgt = torch.tensor([[bos_idx]], dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_mask = self.make_tgt_mask(tgt, pad_idx)
            dec_out  = self.decoder(tgt, enc_out, src_mask, tgt_mask)
            logits   = self.output_projection(dec_out)

            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt      = torch.cat([tgt, next_tok], dim=1)

            if next_tok.item() == eos_idx:
                break

        # ── Decode token ids → English string ────────────────────────────────
        pred_ids = tgt[0, 1:].tolist()
        pred_ids = [t for t in pred_ids if t not in (eos_idx, pad_idx)]

        if hasattr(self, "tgt_vocab") and self.tgt_vocab is not None:
            words = self.tgt_vocab.decode(pred_ids)
        else:
            words = [str(t) for t in pred_ids]

        return " ".join(words)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Noam LR Scheduler  (also lives in lr_scheduler.py)
#    Duplicated here so the autograder can import it from either file.
# ══════════════════════════════════════════════════════════════════════════════

class NoamScheduler:
    """
    Noam learning-rate schedule from "Attention Is All You Need" §5.3:

        lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Linearly increases LR for the first warmup_steps steps, then decreases
    it proportionally to the inverse square root of the step number.

    Args:
        optimizer    : torch.optim.Optimizer  (set base lr=1.0)
        d_model      : model dimension
        warmup_steps : number of warm-up steps (paper default: 4000)
        factor       : global scale multiplier (default 1.0)
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000,
                 factor: float = 1.0):
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self.factor       = factor
        self._step        = 0

    def _get_lr(self, step: int) -> float:
        step = max(step, 1)
        return self.factor * (
            self.d_model ** (-0.5)
            * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )

    def step(self):
        """Advance step counter and update the optimizer's lr."""
        self._step += 1
        lr = self._get_lr(self._step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    @property
    def current_lr(self) -> float:
        return self._get_lr(self._step)

    @property
    def current_step(self) -> int:
        return self._step


# ══════════════════════════════════════════════════════════════════════════════
# 9. Checkpoint loader  (attaches vocab to model so infer(str) works)
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(checkpoint_path: str, device=None):
    """
    Load a saved checkpoint and return a ready-to-use Transformer.

    The src_vocab and tgt_vocab are attached directly to the model so that
    model.infer(german_string) works without any extra arguments.

    Args:
        checkpoint_path : path to the .pt file saved by train.py
        device          : torch.device (auto-detected if None)

    Returns:
        model : Transformer with .src_vocab and .tgt_vocab attached
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt      = torch.load(checkpoint_path, map_location=device,
                           weights_only=False)
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

    # Attach vocabs so infer(string) can tokenise + decode
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab

    return model