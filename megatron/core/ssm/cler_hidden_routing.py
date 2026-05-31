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
    *, value_dim: int, hidden_size: int, dtype, device
) -> nn.Linear:
    """Linear value-space -> hidden-space projection for a routed write residual (zero-init)."""
    proj = nn.Linear(value_dim, hidden_size, bias=False, dtype=dtype, device=device)
    nn.init.zeros_(proj.weight)
    return proj


def project_residual_to_hidden(proj: nn.Linear, residual: Tensor) -> Tensor:
    """Map a write residual [batch, seq, value_heads, value_head_dim] to a hidden-stream
    contribution [seq, batch, hidden] (sequence-first, matching the residual stream layout)."""
    batch, seq = residual.shape[0], residual.shape[1]
    eps = proj(residual.reshape(batch, seq, -1))  # [batch, seq, hidden]
    return eps.transpose(0, 1).contiguous()  # [seq, batch, hidden]
