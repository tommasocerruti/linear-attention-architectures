# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CLER variant of the project-owned pure PyTorch DeltaNet layer.

This module keeps the same projection, convolution, normalization, and output
projection path as :mod:`megatron.core.ssm.delta_net_pytorch`. It only adds the
CLER side channel around the DeltaNet recurrence:

    value_l,t <- value_l,t + gamma_l * residual_prev(l),t

and returns the DeltaNet residual ``v_new`` for the next CLER-capable layer.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.delta_net_pytorch import (
    DeltaNet,
    DeltaNetSubmodules,
    _maybe_compile_linear_rule,
    l2norm,
)
from megatron.core.ssm.gated_delta_net import (
    get_parameter_local_cp,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
)
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push


CLERDeltaNetSubmodules = DeltaNetSubmodules


class CLERDeltaNet(DeltaNet):
    """DeltaNet PyTorch layer with CLER residual reception and emission."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_rule = _maybe_compile_linear_rule(torch_chunk_cler_delta_rule)
        self.supports_cler = True
        self.cler_residual = None
        if self.config.cler_enabled:
            self.cler_gamma = self._make_cler_gamma_parameter()
        else:
            self.register_parameter("cler_gamma", None)

    def _make_cler_gamma_parameter(self):
        """Create the receiver-side CLER gate for this local attention shard."""
        if self.config.cler_gamma_mode == "head":
            if self.cp_size != 1:
                raise NotImplementedError(
                    "CLER per-head gamma is currently implemented for context parallel size 1."
                )
            gamma = torch.full(
                (self.num_heads_local_tp,),
                float(self.config.cler_gamma_init),
                dtype=self.config.params_dtype,
                device=torch.cuda.current_device(),
            )
            parameter = nn.Parameter(gamma)
            setattr(parameter, "tensor_model_parallel", True)
            setattr(parameter, "partition_dim", 0)
            return parameter

        return nn.Parameter(
            torch.tensor(
                self.config.cler_gamma_init,
                dtype=self.config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )

    def _cler_gamma_for_value(self, value: Tensor) -> Tensor:
        gamma = self.cler_gamma.to(dtype=value.dtype)
        if gamma.ndim == 0:
            return gamma
        if gamma.numel() != value.shape[2]:
            raise ValueError(
                "CLER per-head gamma must match the local DeltaNet value-head count, "
                f"got {gamma.numel()=} and value heads={value.shape[2]}."
            )
        return gamma.view(1, 1, -1, 1)

    def _maybe_normalize_cler_residual(self, cler_residual: Tensor) -> Tensor:
        if not self.config.cler_normalize_residual:
            return cler_residual
        residual_fp32 = cler_residual.float()
        scale = torch.rsqrt(
            residual_fp32.square().mean(dim=-1, keepdim=True)
            + self.config.cler_residual_norm_eps
        )
        return (residual_fp32 * scale).to(dtype=cler_residual.dtype)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        del attention_mask, sequence_len_offset
        cler_residual = kwargs.pop("cler_residual", None)
        del kwargs
        self.cler_residual = None
        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size * self.cp_size

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "CLERDeltaNet does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            raise NotImplementedError("CLERDeltaNet does not support inference for now.")

        if packed_seq_params is not None:
            raise NotImplementedError("CLERDeltaNet does not support packed sequence for now.")

        nvtx_range_push(suffix="in_proj")
        qkvb, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        qkvb = tensor_a2a_cp2hp(
            qkvb,
            seq_dim=0,
            head_dim=-1,
            cp_group=self.pg_collection.cp,
            split_sections=[
                self.qk_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
                self.num_heads_local_tp,
            ],
        )

        qkvb = qkvb.transpose(0, 1)
        qkv, beta = torch.split(
            qkvb,
            [
                (2 * self.qk_dim_local_tp + self.v_dim_local_tp) // self.cp_size,
                self.num_heads_local_tp // self.cp_size,
            ],
            dim=-1,
        )
        beta = beta.reshape(batch, seq_len, -1)

        nvtx_range_push(suffix="conv1d")
        seq_len = qkv.shape[1]
        qkv_channels_split_sections = [
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
        ]
        conv1d_weight = get_parameter_local_cp(
            self.conv1d.weight,
            dim=0,
            cp_group=self.pg_collection.cp,
            split_sections=qkv_channels_split_sections,
        )
        conv1d_bias = (
            get_parameter_local_cp(
                self.conv1d.bias,
                dim=0,
                cp_group=self.pg_collection.cp,
                split_sections=qkv_channels_split_sections,
            )
            if self.conv_bias
            else None
        )
        qkv = qkv.transpose(1, 2).contiguous()
        conv_out = F.conv1d(
            input=qkv,
            weight=conv1d_weight,
            bias=conv1d_bias,
            stride=self.conv1d.stride,
            padding=self.conv1d.padding,
            dilation=self.conv1d.dilation,
            groups=self.conv_dim_local_tp // self.cp_size,
        )
        qkv = self.act_fn(conv_out[..., :seq_len])
        qkv = qkv.transpose(1, 2)
        nvtx_range_pop(suffix="conv1d")

        nvtx_range_push(suffix="prepare_qkv_for_delta_rule")
        query, key, value, beta = self._prepare_qkv_for_delta_rule(qkv, beta, batch, seq_len)
        nvtx_range_pop(suffix="prepare_qkv_for_delta_rule")

        if self.config.cler_enabled:
            if cler_residual is not None:
                if self.config.cler_detach_residual:
                    cler_residual = cler_residual.detach()
                if cler_residual.shape != value.shape:
                    raise ValueError(
                        "CLER residual shape must match the current DeltaNet value shape, "
                        f"got {cler_residual.shape=} and {value.shape=}."
                    )
                cler_residual = self._maybe_normalize_cler_residual(cler_residual)
                value = value + self._cler_gamma_for_value(value) * cler_residual
            else:
                value = value + self._cler_gamma_for_value(value) * value.new_zeros(())

        nvtx_range_push(suffix="delta_rule")
        if self.config.cler_enabled:
            core_attn_out, _, self.cler_residual = self.delta_rule(
                query,
                key,
                value,
                beta=beta.sigmoid(),
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                return_residual=True,
            )
        else:
            core_attn_out, _ = self.delta_rule(
                query,
                key,
                value,
                beta=beta.sigmoid(),
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
            )
        nvtx_range_pop(suffix="delta_rule")

        nvtx_range_push(suffix="norm")
        norm_out = self._apply_norm(core_attn_out)
        nvtx_range_pop(suffix="norm")

        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()
        norm_out = tensor_a2a_hp2cp(
            norm_out, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp
        )

        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")
        return out, out_bias


def torch_chunk_cler_delta_rule(
    query,
    key,
    value,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    return_residual=False,
):
    """Torch-native chunked DeltaNet rule with optional CLER residual output."""

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )
    attn = -((k_beta @ key.transpose(-1, -2))).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cum = attn @ k_beta
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    residual_out = torch.zeros_like(value) if return_residual else None
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask, 0)
        v_prime = k_cum[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        if return_residual:
            residual_out[:, :, i] = v_new
        core_attn_out[:, :, i] = q_i @ last_recurrent_state + attn @ v_new
        last_recurrent_state = last_recurrent_state + k_i.transpose(-1, -2) @ v_new

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    if return_residual:
        residual_out = residual_out.reshape(
            residual_out.shape[0], residual_out.shape[1], -1, residual_out.shape[-1]
        )
        residual_out = residual_out[:, :, :sequence_length]
        residual_out = residual_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state, residual_out
    return core_attn_out, last_recurrent_state
