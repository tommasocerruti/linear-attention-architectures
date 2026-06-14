# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure PyTorch additive linear-transformer layer.

This module reuses the project-owned DeltaNet PyTorch shell and swaps only the
recurrent rule. It is intended as the "DeltaNet without the delta rule" baseline:

    S_t = S_{t-1} + k_t^T v_t
    y_t = q_t S_t

The DeltaNet shell still computes the beta projection for architecture-level
comparability with DeltaNet, but this additive-memory rule intentionally ignores
beta.
"""

import torch
import torch.nn.functional as F

from megatron.core.ssm.delta_net_pytorch import (
    DeltaNet,
    DeltaNetSubmodules,
    _maybe_compile_linear_rule,
    l2norm,
)


LinearTransformerSubmodules = DeltaNetSubmodules


class LinearTransformer(DeltaNet):
    """DeltaNet-style layer using pure additive fast-weight memory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_rule = _maybe_compile_linear_rule(torch_chunk_linear_transformer_rule)


def torch_chunk_linear_transformer_rule(
    query,
    key,
    value,
    beta=None,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Torch-native chunked additive linear-memory rule.

    Inputs follow the DeltaNet PyTorch convention: ``[batch, sequence, heads, dim]``.
    The returned output has the same shape as ``value``.
    """

    del beta

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    query, key, value = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value)
    ]

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask, 0)
        core_attn_out[:, :, i] = q_i @ last_recurrent_state + attn @ v_i
        last_recurrent_state = last_recurrent_state + k_i.transpose(-1, -2) @ v_i

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
