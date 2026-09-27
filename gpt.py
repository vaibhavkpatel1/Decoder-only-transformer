"""Decoder-only Transformer (GPT-style) in PyTorch.

No encoder and no cross-attention: every block is masked self-attention
plus a feed-forward net. Q, K, and V all come from the same token sequence.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """One GPT block: causal self-attention + feed-forward (no cross-attention)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # [B, H, T, D]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq, d_k = x.shape
        x = x.transpose(1, 2).contiguous()  # [B, T, H, D]
        return x.view(batch, seq, heads * d_k)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = GPT.layer_normalisation(x, self.ln1)
        q = self._split_heads(GPT.linear(h, self.w_q))
        k = self._split_heads(GPT.linear(h, self.w_k))
        v = self._split_heads(GPT.linear(h, self.w_v))
        attn = GPT.attention(q, k, v, mask)
        attn = GPT.linear(self._merge_heads(attn), self.w_o)
        x = x + self.dropout(attn)

        h = GPT.layer_normalisation(x, self.ln2)
        x = x + self.dropout(GPT.feed_forward(h, self.fc1, self.fc2))
        return x


class GPT(nn.Module):
    """Decoder-only language model.

    Methods below are the building blocks. Forward path:

        tokens -> embedding * sqrt(d_model) + positional encoding
               -> N decoder blocks (mask + attention + LN + FFN)
               -> final LN -> linear (vocab)  [softmax is optional, for probs]
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 64,
        max_len: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos", self.positional_encoding(max_len, d_model), persistent=False)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    @staticmethod
    def linear(x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        return layer(x)

    @staticmethod
    def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return F.softmax(x, dim=dim)

    @staticmethod
    def layer_normalisation(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        return norm(x)

    def positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Fixed sinusoidal encodings (Vaswani et al.). Shape [1, T, D]."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def masking(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Causal (look-ahead) mask for decoder self-attention.

        Additive mask: 0 where token j may attend to i (j <= i), -inf for
        future tokens. Shape [1, 1, T, T] so it broadcasts over batch and heads.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf")).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scaled dot-product attention.

        query, key, value: [B, H, T, d_k]. In a decoder-only model they are
        three projections of the *same* sequence (not encoder memory).
        """
        d_k = query.size(-1)
        scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            scores = scores + mask
        weights = GPT.softmax(scores, dim=-1)
        return weights @ value

    @staticmethod
    def feed_forward(x: torch.Tensor, fc1: nn.Linear, fc2: nn.Linear) -> torch.Tensor:
        return GPT.linear(F.relu(GPT.linear(x, fc1)), fc2)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return vocab logits of shape [B, T, vocab]."""
        seq = token_ids.size(1)
        x = self.embed(token_ids) * math.sqrt(self.d_model) + self.pos[:, :seq]
        x = self.dropout(x)
        mask = self.masking(seq, token_ids.device)
        for block in self.blocks:
            x = block(x, mask)
        x = self.layer_normalisation(x, self.ln_f)
        return self.linear(x, self.lm_head)

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        eos_id: int | None = None,
        max_new: int = 16,
    ) -> torch.Tensor:
        """Greedy decode. `prefix` is [B, T] token ids; returns [B, T + new]."""
        self.eval()
        out = prefix
        for _ in range(max_new):
            logits = self(out[:, -self.max_len :])
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return out
