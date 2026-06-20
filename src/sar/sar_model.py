"""
Schema-Aware Representation (SAR) model architecture.
Adapted from SchemaRAG train_SAR.py (SchemaAwareModel + SafeMultiheadAttention):
- Architecture preserved exactly — two cross-attention layers:
    1. table_column_attention: column embeddings attend to their table
    2. question_table_attention: question attends to column-aware tables
- NaN/Inf guards from SafeMultiheadAttention kept (critical for stable training).
- FlagEmbedding (BGE) used for encoding question / tables / columns.
- embed_dim=1024 matches BAAI/bge-large-en-v1.5 output dimension.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.embed_dim = embed_dim

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=-1)
            if all_masked.any():
                attn_output = torch.zeros_like(query)
                valid = ~all_masked
                if valid.any():
                    try:
                        out, w = self.attention(
                            query[valid], key[valid], value[valid],
                            key_padding_mask=key_padding_mask[valid],
                            attn_mask=attn_mask,
                        )
                        attn_output[valid] = out
                        return attn_output, w
                    except Exception:
                        pass
                return attn_output, None

        try:
            return self.attention(query, key, value,
                                  key_padding_mask=key_padding_mask,
                                  attn_mask=attn_mask)
        except Exception:
            return query.clone(), None


class SchemaAwareModel(nn.Module):
    """
    Two-stage cross-attention model for schema-aware query encoding.

    Input embeddings are produced by FlagModel (BGE-large, dim=1024).
    Output is a single vector per (question, schema) pair used as the
    retrieval key for SAR.
    """

    def __init__(self, embed_dim: int = 1024, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.table_column_attention   = SafeMultiheadAttention(embed_dim, num_heads, dropout)
        self.question_table_attention = SafeMultiheadAttention(embed_dim, num_heads, dropout)

        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        self.table_proj    = nn.Linear(embed_dim, embed_dim)
        self.column_proj   = nn.Linear(embed_dim, embed_dim)
        self.question_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj   = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _safe_norm(
        self,
        x: torch.Tensor,
        norm: nn.LayerNorm,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        try:
            inp = x + residual if residual is not None else x
            if torch.isnan(inp).any() or torch.isinf(inp).any():
                return residual if residual is not None else torch.zeros_like(x)
            out = norm(inp)
            if torch.isnan(out).any() or torch.isinf(out).any():
                return residual if residual is not None else torch.zeros_like(x)
            return out
        except Exception:
            return residual if residual is not None else torch.zeros_like(x)

    def forward(
        self,
        question_embed: torch.Tensor,       # [B, D]
        table_embeds:   torch.Tensor,       # [B, T, D]
        column_embeds:  torch.Tensor,       # [B, T, C, D]
        table_masks:    torch.Tensor,       # [B, T]  True = valid
        column_masks:   torch.Tensor,       # [B, T, C]
    ) -> torch.Tensor:                      # [B, D]
        B, T, D = table_embeds.shape
        C       = column_embeds.shape[2]

        # Stage 1: for each table, attend over its columns → column-aware table embed
        col_aware_tables = []
        for i in range(T):
            t_i   = table_embeds[:, i:i+1, :]          # [B, 1, D]
            cols_i = column_embeds[:, i, :, :]          # [B, C, D]
            mask_i = column_masks[:, i, :]              # [B, C]

            has_cols = mask_i.any(dim=-1)
            t_proj   = self.dropout(self.table_proj(t_i))
            c_proj   = self.dropout(self.column_proj(cols_i))

            if has_cols.any():
                out, _ = self.table_column_attention(
                    t_proj, c_proj, c_proj,
                    key_padding_mask=~mask_i.bool(),
                )
                col_aware = self._safe_norm(out, self.layer_norm1, t_proj)
            else:
                col_aware = t_proj

            # For rows without any valid column, fall back to original table embed
            col_aware = torch.where(
                has_cols.unsqueeze(1).unsqueeze(2).expand_as(col_aware),
                col_aware, t_proj,
            )
            col_aware_tables.append(col_aware)

        col_aware_table_embeds = torch.cat(col_aware_tables, dim=1)  # [B, T, D]

        # Stage 2: question attends over column-aware tables
        q_proj = self.dropout(self.question_proj(question_embed.unsqueeze(1)))  # [B, 1, D]
        out, _ = self.question_table_attention(
            q_proj, col_aware_table_embeds, col_aware_table_embeds,
            key_padding_mask=~table_masks.bool(),
        )
        schema_aware = self._safe_norm(out, self.layer_norm2, q_proj)
        schema_aware = schema_aware.squeeze(1)                        # [B, D]

        return self.output_proj(schema_aware)
