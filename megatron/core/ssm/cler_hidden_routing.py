# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CLER-H: route the GDN delta-rule write residual into the SHARED hidden/residual stream.

Standard CLER injects the write residual r_l into the next layer's value target, which lives in a
layer-specific value-projection space (misaligned across layers). CLER-H instead projects r_l into
the model hidden space (d_model) -- the same aligned space the residual stream / Attention Residuals
operate in -- and adds the routed error to the hidden state entering later layers:

    eps_l = P_l(r_l) in R^{d_model}        (P_l learned, zero-init -> starts as a no-op = baseline)
    h_in_l <- h_in_l + eps_{prev GDN}      ("latest" routing; carried across non-GDN layers)

This keeps CLER's signal (the delta-rule error) but fixes the injection SPACE. It differs from
AttnRes, which routes layer OUTPUTS (not errors) and replaces (not adds). Zero-init projection means
the model begins exactly at the GDN baseline and learns whether/how much error to route.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def make_cler_hidden_projection(
    *, value_dim: int, hidden_size: int, dtype, device, rank: int = 0
) -> nn.Module:
    """Value-space -> hidden-space projection for a routed write residual.

    rank == 0: a full dense Linear(value_dim -> hidden_size) (large: value_dim*hidden_size params).
    rank  > 0: a low-rank bottleneck Linear(value_dim -> rank) -> Linear(rank -> hidden_size), with
    far fewer params (value_dim*rank + rank*hidden_size). Output weight zero-init either way, so the
    routed contribution starts at 0 (exact baseline) and is learned.
    """
    if rank and rank > 0:
        down = nn.Linear(value_dim, rank, bias=False, dtype=dtype, device=device)
        up = nn.Linear(rank, hidden_size, bias=False, dtype=dtype, device=device)
        nn.init.zeros_(up.weight)  # start at zero contribution
        return nn.Sequential(down, up)
    proj = nn.Linear(value_dim, hidden_size, bias=False, dtype=dtype, device=device)
    nn.init.zeros_(proj.weight)
    return proj


def project_residual_to_hidden(
    proj: nn.Linear, residual: Tensor, normalize: bool = False, eps: float = 1e-6
) -> Tensor:
    """Map a write residual [batch, seq, value_heads, value_head_dim] to a hidden-stream
    contribution [seq, batch, hidden] (sequence-first, matching the residual stream layout).

    If ``normalize`` is set, the flattened per-token routed vector is RMS-normalized to unit RMS
    before the projection (magnitude control: makes value v and residual r enter at identical scale)."""
    batch, seq = residual.shape[0], residual.shape[1]
    x = residual.reshape(batch, seq, -1)  # [batch, seq, value_dim]
    if normalize:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        x = xf.to(residual.dtype)
    out = proj(x)  # [batch, seq, hidden]
    return out.transpose(0, 1).contiguous()  # [seq, batch, hidden]
