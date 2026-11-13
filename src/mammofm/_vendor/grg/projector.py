"""Four-view attentional projector used by Mammo-GRG.

Derived from the CoCa attentional pooler and the four-view LLaVA adaptation
used for the released Mammo-GRG checkpoints.
"""

from __future__ import annotations

import re

import torch
from einops import rearrange, repeat
from torch import einsum, nn


class AttentionalPooler(nn.Module):
    def __init__(self, d_model: int, context_dim: int, n_head: int = 8, n_queries: int = 512):
        super().__init__()
        if d_model % n_head:
            raise ValueError(f"d_model={d_model} must be divisible by n_head={n_head}")
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.scale = (d_model // n_head) ** -0.5
        self.heads = n_head
        self.ln_k = nn.LayerNorm(context_dim)
        self.ln_q = nn.LayerNorm(d_model)
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_kv = nn.Linear(context_dim, d_model * 2, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = rearrange(x, "b n d -> b 1 n d")
        q = repeat(self.query, "n d -> b m n d", b=x.shape[0], m=x.shape[1])
        x = self.ln_k(x)
        q = self.to_q(self.ln_q(q))
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q = rearrange(q, "b t n (h d) -> b h t n d", h=self.heads) * self.scale
        k = rearrange(k, "b t n (h d) -> b h t n d", h=self.heads)
        v = rearrange(v, "b t n (h d) -> b h t n d", h=self.heads)
        similarity = einsum("... i d, ... j d -> ... i j", q, k)
        similarity = similarity - similarity.amax(dim=-1, keepdim=True).detach()
        attention = similarity.softmax(dim=-1)
        output = einsum("... i j, ... j d -> ... i d", attention, v)
        output = rearrange(output, "b h t n d -> b t n (h d)")
        return self.to_out(output).squeeze(dim=1)


class AttentionalPoolProjector(nn.Module):
    def __init__(self, embed_dim: int, context_dim: int, n_queries: int, projector: nn.Module):
        super().__init__()
        self.attn_pool = AttentionalPooler(
            d_model=embed_dim, context_dim=context_dim, n_head=8, n_queries=n_queries
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.proj = projector
        output_dim = projector[-1].out_features
        self.view_positional_embeddings = nn.Parameter(torch.randn(4, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.ln(self.attn_pool(x)))


def build_vision_projector(config) -> nn.Module:
    projector_type = getattr(config, "mm_projector_type", "attn_pool+mlp2x_gelu")
    if not projector_type.startswith("attn_pool+"):
        raise ValueError(
            "Mammo-GRG checkpoints require an attentional-pool projector; "
            f"received {projector_type!r}."
        )
    mlp_match = re.fullmatch(r"mlp(\d+)x_gelu", projector_type.split("+", 1)[1])
    if not mlp_match:
        raise ValueError(f"Unsupported projector: {projector_type}")
    depth = int(mlp_match.group(1))
    modules: list[nn.Module] = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
    for _ in range(1, depth):
        modules.extend((nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)))
    return AttentionalPoolProjector(
        embed_dim=config.mm_hidden_size,
        context_dim=config.mm_context_size,
        n_queries=config.mm_n_queries,
        projector=nn.Sequential(*modules),
    )
