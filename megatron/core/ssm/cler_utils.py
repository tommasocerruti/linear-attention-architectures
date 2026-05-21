# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared CLER helpers for Megatron SSM attention variants."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

try:
    from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu,
        chunk_gated_delta_rule_fwd_h,
    )
    from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
    from fla.ops.cp.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu_pre_process,
        chunk_gated_delta_rule_fwd_h_pre_process,
        compress_h0,
        expand_h0,
    )
    from fla.ops.delta_rule import chunk_delta_rule
    from fla.ops.delta_rule.wy_fast import (
        prepare_wy_repr_bwd as prepare_delta_wy_repr_bwd,
    )
    from fla.ops.delta_rule.wy_fast import (
        prepare_wy_repr_fwd as prepare_delta_wy_repr_fwd,
    )
    from fla.ops.delta_rule.wy_fast import (
        recompute_w_u_fwd as recompute_delta_w_u_fwd,
    )
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.gated_delta_rule.gate import (
        gdn_gate_bwd,
        gdn_gate_chunk_cumsum,
    )
    from fla.ops.gated_delta_rule.wy_fast import (
        prepare_wy_repr_bwd as prepare_gated_wy_repr_bwd,
    )
    from fla.ops.gated_delta_rule.wy_fast import (
        recompute_w_u_fwd as recompute_gated_w_u_fwd,
    )
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.index import prepare_chunk_indices
    from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

    HAVE_FLA = True
except ImportError:
    chunk_delta_rule = None
    chunk_gated_delta_rule = None

    def input_guard(fn=None, **_kwargs):
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    def autocast_custom_fwd(fn):
        return fn

    def autocast_custom_bwd(fn):
        return fn

    HAVE_FLA = False


def make_cler_gamma_parameter(
    *,
    config,
    num_value_heads_local_tp: int,
    value_head_dim: int,
    cp_size: int,
    variant_name: str,
) -> nn.Parameter:
    """Create the receiver-side CLER gate for one local attention shard."""

    if config.cler_gamma_mode == "head":
        if cp_size != 1:
            raise NotImplementedError(
                f"CLER per-head gamma for {variant_name} is currently implemented "
                "for context parallel size 1."
            )
        gamma = torch.full(
            (num_value_heads_local_tp,),
            float(config.cler_gamma_init),
            dtype=config.params_dtype,
            device=torch.cuda.current_device(),
        )
        parameter = nn.Parameter(gamma)
        setattr(parameter, "tensor_model_parallel", True)
        setattr(parameter, "partition_dim", 0)
        return parameter

    if config.cler_gamma_mode == "channel":
        if cp_size != 1:
            raise NotImplementedError(
                f"CLER per-channel gamma for {variant_name} is currently implemented "
                "for context parallel size 1."
            )
        gamma = torch.full(
            (num_value_heads_local_tp, value_head_dim),
            float(config.cler_gamma_init),
            dtype=config.params_dtype,
            device=torch.cuda.current_device(),
        )
        parameter = nn.Parameter(gamma)
        setattr(parameter, "tensor_model_parallel", True)
        setattr(parameter, "partition_dim", 0)
        return parameter

    return nn.Parameter(
        torch.tensor(
            config.cler_gamma_init,
            dtype=config.params_dtype,
            device=torch.cuda.current_device(),
        )
    )


def cler_gamma_for_value(gamma: Tensor, value: Tensor, variant_name: str) -> Tensor:
    """Broadcast a scalar/head/channel CLER gamma to a value tensor."""

    gamma = gamma.to(dtype=value.dtype)
    if gamma.ndim == 0:
        return gamma
    if gamma.ndim == 1:
        if gamma.numel() != value.shape[2]:
            raise ValueError(
                f"CLER per-head gamma must match the local {variant_name} value-head count, "
                f"got gamma.numel()={gamma.numel()} and value heads={value.shape[2]}."
            )
        return gamma.view(1, 1, -1, 1)
    if gamma.ndim == 2:
        if tuple(gamma.shape) != tuple(value.shape[2:4]):
            raise ValueError(
                f"CLER per-channel gamma must match the local {variant_name} value shape "
                f"per token, got gamma shape={tuple(gamma.shape)} and "
                f"value shape={tuple(value.shape[2:4])}."
            )
        return gamma.view(1, 1, gamma.shape[0], gamma.shape[1])
    raise ValueError(
        "CLER gamma must be scalar, per-head, or per-channel, "
        f"got shape={tuple(gamma.shape)}."
    )


def normalize_cler_residual(config, cler_residual: Tensor) -> Tensor:
    """Optionally RMS-normalize a routed CLER residual."""

    if not config.cler_normalize_residual:
        return cler_residual
    residual_fp32 = cler_residual.float()
    scale = torch.rsqrt(
        residual_fp32.square().mean(dim=-1, keepdim=True) + config.cler_residual_norm_eps
    )
    return (residual_fp32 * scale).to(dtype=cler_residual.dtype)


def inject_cler_residual(
    *,
    value: Tensor,
    cler_residual: Tensor | None,
    cler_gamma: Tensor,
    config,
    variant_name: str,
) -> Tensor:
    """Inject a previous layer's CLER residual into the current value tensor."""

    gamma = cler_gamma_for_value(cler_gamma, value, variant_name)
    if cler_residual is None:
        return value + gamma * value.new_zeros(())

    if config.cler_detach_residual:
        cler_residual = cler_residual.detach()
    if cler_residual.shape != value.shape:
        raise ValueError(
            f"CLER residual shape must match the current {variant_name} value shape, "
            f"got cler_residual.shape={cler_residual.shape} and value.shape={value.shape}."
        )
    cler_residual = normalize_cler_residual(config, cler_residual)
    return value + gamma * cler_residual


def _require_fla() -> None:
    if not HAVE_FLA:
        raise ImportError(
            "FLA is not installed. Please install flash-linear-attention to use fast CLER."
        )


def _add_residual_grad(dv: Tensor, d_residual: Tensor | None) -> Tensor:
    if d_residual is not None:
        dv = dv.add(d_residual.to(dtype=dv.dtype))
    return dv


class _CLERChunkDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        beta: Tensor,
        scale: float,
        initial_state: Tensor | None,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: Tensor | None = None,
        cu_seqlens_cpu: Tensor | None = None,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens_cpu)
            if cu_seqlens is not None
            else None
        )
        w, u, A = prepare_delta_wy_repr_fwd(
            k=k,
            v=v,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=None,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        o = chunk_fwd_o(
            q=q,
            k=k,
            v=v_new,
            h=h,
            g=None,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, beta, A, initial_state, cu_seqlens, chunk_indices)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state, v_new.to(v.dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: Tensor,
        dht: Tensor | None,
        d_residual: Tensor | None,
    ):
        q, q_rstd, k, k_rstd, v, beta, A, initial_state, cu_seqlens, chunk_indices = (
            ctx.saved_tensors
        )
        w, u = recompute_delta_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=None,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        if do is None:
            do = torch.zeros_like(v_new)
        dv = chunk_bwd_dv_local(
            q=q,
            k=k,
            do=do,
            g=None,
            scale=ctx.scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        dv = _add_residual_grad(dv, d_residual)
        dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
            q=q,
            k=k,
            w=w,
            g=None,
            h0=initial_state,
            dht=dht,
            do=do,
            dv=dv,
            scale=ctx.scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        dq, dk, dw, _ = chunk_bwd_dqkwg(
            q=q,
            k=k,
            v=v_new,
            h=h,
            w=w,
            dv=dv,
            do=do,
            dh=dh,
            g=None,
            scale=ctx.scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        dk2, dv, db = prepare_delta_wy_repr_bwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            dw=dw,
            du=dv,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        dk.add_(dk2)
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), None, dh0, None, None, None, None


class _CLERChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        g: Tensor,
        beta: Tensor,
        scale: float,
        initial_state: Tensor | None,
        output_final_state: bool,
        cu_seqlens: Tensor | None = None,
        cu_seqlens_cpu: Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        cp_context: Any | None = None,
        transpose_state_layout: bool = False,
        use_gate_in_kernel: bool = False,
        A_log: Tensor | None = None,
        dt_bias: Tensor | None = None,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens_cpu)
            if cu_seqlens is not None
            else None
        )
        g_input = g if use_gate_in_kernel else None
        if use_gate_in_kernel:
            g = gdn_gate_chunk_cumsum(
                g=g,
                A_log=A_log,
                chunk_size=64,
                scale=RCP_LN2,
                dt_bias=dt_bias,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        else:
            g = chunk_local_cumsum(
                g,
                chunk_size=64,
                scale=RCP_LN2,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )

        w, u, A = chunk_gated_delta_rule_fwd_intra(
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=True,
        )

        if cp_context is not None:
            initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
                k=k,
                w=w,
                u=u,
                g=g,
                cu_seqlens=cu_seqlens,
                initial_state=initial_state,
                context=cp_context,
                use_exp2=True,
                transpose_state_layout=transpose_state_layout,
            )

        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=True,
            transpose_state_layout=transpose_state_layout,
        )

        if cp_context is not None:
            initial_state = compress_h0(initial_state, context=cp_context)

        o = chunk_fwd_o(
            q=q,
            k=k,
            v=v_new,
            h=h,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=True,
            transpose_state_layout=transpose_state_layout,
        )
        ctx.save_for_backward(
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            g,
            beta,
            A,
            initial_state,
            cu_seqlens,
            chunk_indices,
            g_input,
            A_log,
            dt_bias,
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.cp_context = cp_context
        ctx.transpose_state_layout = transpose_state_layout
        ctx.use_gate_in_kernel = use_gate_in_kernel
        return o.to(q.dtype), final_state, v_new.to(v.dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: Tensor,
        dht: Tensor | None,
        d_residual: Tensor | None,
    ):
        (
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            g,
            beta,
            A,
            initial_state,
            cu_seqlens,
            chunk_indices,
            g_input,
            A_log,
            dt_bias,
        ) = ctx.saved_tensors

        w, u = recompute_gated_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=True,
        )

        if ctx.cp_context is not None:
            initial_state = expand_h0(initial_state, context=ctx.cp_context)

        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=True,
            transpose_state_layout=ctx.transpose_state_layout,
        )
        if do is None:
            do = torch.zeros_like(v_new)
        dv = chunk_bwd_dv_local(
            q=q,
            k=k,
            g=g,
            do=do,
            scale=ctx.scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=True,
        )
        dv = _add_residual_grad(dv, d_residual)

        if ctx.cp_context is not None:
            dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(
                q=q,
                k=k,
                w=w,
                do=do,
                dv=dv,
                g=g,
                scale=ctx.scale,
                cu_seqlens=cu_seqlens,
                dht=dht,
                initial_state=initial_state,
                context=ctx.cp_context,
                use_exp2=True,
                transpose_state_layout=ctx.transpose_state_layout,
            )

        dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
            q=q,
            k=k,
            w=w,
            g=g,
            h0=initial_state,
            dht=dht,
            do=do,
            dv=dv,
            scale=ctx.scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=True,
            transpose_state_layout=ctx.transpose_state_layout,
        )
        dq, dk, dw, dg = chunk_bwd_dqkwg(
            q=q,
            k=k,
            v=v_new,
            w=w,
            g=g,
            h=h,
            dv=dv,
            do=do,
            dh=dh,
            scale=ctx.scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=True,
            transpose_state_layout=ctx.transpose_state_layout,
        )
        dk2, dv, db, dg2 = prepare_gated_wy_repr_bwd(
            k=k,
            v=v,
            beta=beta,
            g=g,
            A=A,
            dw=dw,
            du=dv,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=True,
        )
        dk.add_(dk2)
        dg.add_(dg2)
        dg = chunk_local_cumsum(
            dg,
            chunk_size=64,
            reverse=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        dA_log, ddt_bias = None, None
        if ctx.use_gate_in_kernel:
            dg, dA_log, ddt_bias = gdn_gate_bwd(
                g=g_input, A_log=A_log, dt_bias=dt_bias, dyg=dg
            )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            dh0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dA_log,
            ddt_bias,
        )


def chunk_delta_rule_with_residual(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    scale: float | None = None,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Tensor | None = None,
    cu_seqlens_cpu: Tensor | None = None,
    return_residual: bool = False,
    **kwargs,
):
    """FLA chunk DeltaNet rule with optional CLER residual output."""

    _require_fla()
    if not return_residual:
        return chunk_delta_rule(
            q=q,
            k=k,
            v=v,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            **kwargs,
        )

    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, (
        "CLERChunkDeltaRuleFunction does not support float32. Please use bfloat16."
    )
    assert len(beta.shape) == 3, "beta must be of shape [batch, seq_len, heads]."
    if "head_first" in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} "
                "when using `cu_seqlens`. Please flatten variable-length inputs "
                "before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                "The number of initial states is expected to equal the number of "
                f"input sequences, got {initial_state.shape[0]} and {len(cu_seqlens) - 1}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return _CLERChunkDeltaRuleFunction.apply(
        q,
        k,
        v,
        beta,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        cu_seqlens_cpu,
    )


def chunk_gated_delta_rule_with_residual(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    scale: float | None = None,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Tensor | None = None,
    cu_seqlens_cpu: Tensor | None = None,
    cp_context: Any | None = None,
    transpose_state_layout: bool = False,
    return_residual: bool = False,
    **kwargs,
):
    """FLA chunk Gated DeltaNet rule with optional CLER residual output."""

    _require_fla()
    if not return_residual:
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cp_context=cp_context,
            transpose_state_layout=transpose_state_layout,
            **kwargs,
        )

    if q.shape[2] != k.shape[2]:
        raise ValueError(
            "q and k must have the same number of heads, "
            f"but got q.shape[2]={q.shape[2]} and k.shape[2]={k.shape[2]}"
        )
    num_key_heads, num_value_heads = q.shape[2], v.shape[2]
    if num_value_heads % num_key_heads != 0:
        raise ValueError(
            f"For GVA, num_v_heads ({num_value_heads}) must be evenly divisible by "
            f"num_heads ({num_key_heads}), got remainder "
            f"{num_value_heads % num_key_heads}."
        )
    if "head_first" in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )

    if cp_context is not None:
        assert initial_state is None, "Initial state is not supported for CP"
        assert output_final_state is False, "Output final state is not supported for CP"
        assert cp_context.cu_seqlens is not None, "cu_seqlens is required for CP"
        cu_seqlens = cp_context.cu_seqlens
        if cp_context.cu_seqlens_cpu is not None:
            cu_seqlens_cpu = cp_context.cu_seqlens_cpu

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} "
                "when using `cu_seqlens`. Please flatten variable-length inputs "
                "before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                "The number of initial states is expected to equal the number of "
                f"input sequences, got {initial_state.shape[0]} and {len(cu_seqlens) - 1}."
            )

    use_gate_in_kernel = kwargs.get("use_gate_in_kernel", False)
    A_log = kwargs.get("A_log")
    dt_bias = kwargs.get("dt_bias")
    if use_gate_in_kernel:
        assert A_log is not None, "A_log must be provided when use_gate_in_kernel=True."

    if scale is None:
        scale = k.shape[-1] ** -0.5
    return _CLERChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
        use_qk_l2norm_in_kernel,
        cp_context,
        transpose_state_layout,
        use_gate_in_kernel,
        A_log,
        dt_bias,
    )
