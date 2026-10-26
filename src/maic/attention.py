"""Pre-norm multi-head attention building blocks."""

import torch
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *, padding_mask=None, causal=False):
        normalized = self.norm1(x)
        mask = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).triu(1) if causal else None
        attended = self.attention(normalized, normalized, normalized, key_padding_mask=padding_mask,
                                  attn_mask=mask, need_weights=False)[0]
        x = x + self.dropout(attended)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x.masked_fill(padding_mask[..., None], 0) if padding_mask is not None else x


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.self_block = AttentionBlock(dim, heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, padding_mask, memory_padding_mask):
        x = self.self_block(x, padding_mask=padding_mask, causal=True)
        attended = self.cross_attention(self.norm(x), memory, memory,
                                        key_padding_mask=memory_padding_mask, need_weights=False)[0]
        return (x + self.dropout(attended)).masked_fill(padding_mask[..., None], 0)


class AttentionPool(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, nodes, agent_mask):
        query = self.query.expand(nodes.shape[0], -1, -1)
        return self.norm(self.attention(query, nodes, nodes, key_padding_mask=~agent_mask, need_weights=False)[0][:, 0])
