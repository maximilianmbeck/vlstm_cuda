# -*- coding: utf-8 -*-

# GLA chunkwise recurrent implementation
# Copy from https://github.com/sustcsonglin/flash-linear-attention

import torch
from einops import rearrange

# B: batch size
# NH: number of heads
# S: sequence length
# C: chunk size
# NC: number of chunks
# D: hidden dimension


def torch_simple_gla(q, k, v, g, chunk_size=64):
    q = rearrange(q, "b h (n c) d -> b h n c d", c=chunk_size) * (q.shape[-1] ** -0.5)
    k = rearrange(k, "b h (n c) d -> b h n c d", c=chunk_size)
    v = rearrange(v, "b h (n c) d -> b h n c d", c=chunk_size)  # (B, NH, NC, C, D)
    g = rearrange(g, "b h (n c) -> b h n c", c=chunk_size)
    g = g.cumsum(-1)

    # Note on (-g + g[:, :, :, -1, None]).exp()[..., None] Shape: (B, NH, C, NC, 1):
    # this is broadcasted alongthe D dimension of v
    kv = k.transpose(-1, -2) @ (v * (-g + g[:, :, :, -1, None]).exp()[..., None])
    S = torch.zeros_like(kv)

    for i in range(1, g.shape[-2]):
        S[:, :, i] = (
            S[:, :, i - 1].clone() * g[:, :, i - 1, -1, None, None].exp()
            + kv[:, :, i - 1]
        )

    inter = (q * g[..., None].exp()) @ S
    attn = q @ k.transpose(-1, -2)  # (B, NH, NC, C, C)
    attn = attn * (g[..., None] - g[..., None, :]).exp()
    attn = attn.masked_fill(
        torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1
        ),
        0,
    )
    intra = attn @ v
    o = inter + intra  # (B, NH, NC, C, D)
    y = rearrange(
        o, "b h n c d -> b h (n c) d"
    )  # (B, NH, NC, C, D) -> (B, NH, NC * C, D) = (B, NH, S, D)
    return y


def torch_simple_gla_recurrent(q, k, v, g, chunk_size=64):
    # q = rearrange(q, "b h (n c) d -> b h n c d", c=chunk_size) * (q.shape[-1] ** -0.5)
    # k = rearrange(k, "b h (n c) d -> b h n c d", c=chunk_size)
    # v = rearrange(v, "b h (n c) d -> b h n c d", c=chunk_size)
    # g = rearrange(g, "b h (n c) -> b h n c", c=chunk_size)
    # g = g.cumsum(-1)
    # kv = k.transpose(-1, -2) @ v

    B, H, T, DK = q.shape
    q = q * (DK**-0.5)
    _, _, _, DV = v.shape
    S = torch.zeros(B, H, DK, DV).to(q)
    o = torch.zeros(B, H, T, DV).to(q)
    for i in range(T):
        gate = g[:, :, i].exp()
        key = k[:, :, i]
        value = v[:, :, i]
        kv = key.unsqueeze(-1) * value.unsqueeze(-2)
        S = S.clone() * gate.unsqueeze(-1).unsqueeze(-1) + kv
        q_i = q[:, :, i, :]
        o_i = (q_i.unsqueeze(-1) * S).sum(-2)
        o[:, :, i] = o_i

    return o
