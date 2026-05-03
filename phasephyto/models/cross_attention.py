"""
Cross-Attention Fusion: structural tokens attend to semantic tokens.

The core innovation of PhasePhyto.  Phase Congruency structural tokens
serve as Queries (Q), while the backbone's semantic tokens serve as Keys (K)
and Values (V).  This forces the network to attend to semantic features
*only* at locations where physics-based phase congruency has confirmed
invariant structural boundaries -- explicitly ignoring shadow pseudo-edges.

Target parameter budget: ~331K trainable parameters in the fusion module.
"""

import torch
import torch.nn as nn


class StructuralSemanticFusion(nn.Module):
    """Multi-head cross-attention fusing structural and semantic streams.

    Q = Structural Tokens (from PC Encoder, e.g. 49 tokens)
    K, V = Semantic Tokens (from ViT backbone, e.g. 196 tokens)

    The output has the same sequence length as Q, then globally pooled
    for classification.

    Args:
        fusion_dim: Dimension of both token types (must match).
        num_heads: Number of attention heads.
        dropout: Dropout on attention weights.
        use_residual: Add residual connection from Q to output.
        ffn_hidden_dim: Hidden dimension for the post-attention feed-forward block.
            Defaults to ``fusion_dim // 2`` to keep fusion near the intended
            lightweight parameter budget.
    """

    def __init__(
        self,
        fusion_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_residual: bool = True,
        ffn_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        hidden_dim = ffn_hidden_dim or max(fusion_dim // 2, num_heads)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_q = nn.LayerNorm(fusion_dim)
        self.norm_kv = nn.LayerNorm(fusion_dim)
        self.norm_out = nn.LayerNorm(fusion_dim)

        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(fusion_dim)

    def forward(
        self,
        structural_tokens: torch.Tensor,
        semantic_tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fuse structural and semantic token streams.

        Args:
            structural_tokens: (B, Nq, D) from PC Encoder.
            semantic_tokens: (B, Nkv, D) from backbone.
            return_attention: If True, return attention weight matrix.

        Returns:
            fused: (B, D) global feature vector (mean-pooled).
            attn_weights: (B, Nq, Nkv) or None.
        """
        q = self.norm_q(structural_tokens)  # (B, Nq, D)
        kv = self.norm_kv(semantic_tokens)  # (B, Nkv, D)

        attn_out, attn_weights = self.cross_attn(
            query=q, key=kv, value=kv,
            need_weights=return_attention,
        )  # attn_out: (B, Nq, D)
        if attn_weights is not None:
            attn_weights = attn_weights / attn_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(attn_weights.dtype).tiny)

        if self.use_residual:
            attn_out = attn_out + structural_tokens
        attn_out = self.norm_out(attn_out)

        ffn_out = self.ffn(attn_out)
        fused_tokens = self.norm_ffn(ffn_out + attn_out)  # (B, Nq, D)

        fused = fused_tokens.mean(dim=1)

        return fused, attn_weights
