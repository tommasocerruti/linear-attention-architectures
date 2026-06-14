# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Project-owned Gated DeltaNet-2 layer for CLER experiments.

This module follows the project-owned PyTorch GatedDeltaNet wrapper, but
replaces the scalar beta gate with the Gated DeltaNet-2 channel-wise erase and
write gates:

    S_t = (I - k_t (b_t * k_t)^T) Diag(alpha_t) S_{t-1}
          + k_t (w_t * v_t)^T
    y_t = S_t^T q_t

For efficient training, the default path calls the GatedDeltaNet-2 repository's
external Triton chunk kernel through ``GDN2_REPO_DIR`` without vendoring that
source into this repository. A simple token-serial PyTorch fallback is kept for
shape/debug checks, but it is not suitable for full training.
"""

import logging
import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.jit import jit_fuser
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import (
    _split_tensor_factory,
    get_parameter_local_cp,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
)
from megatron.core.ssm.gated_delta_net_pytorch import l2norm
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_external_chunk_gdn2():
    """Load the GatedDeltaNet-2 repo's chunk kernel without importing lit_gpt.__init__."""

    repo_dir = os.environ.get("GDN2_REPO_DIR")
    if repo_dir is None:
        repo_dir = str(Path(__file__).resolve().parents[4] / "GatedDeltaNet-2")
    ops_parent = Path(repo_dir) / "lit_gpt"
    if not (ops_parent / "gdn2_ops" / "chunk_gdn2.py").is_file():
        raise ImportError(
            "GDN2 chunk kernel not found. Set GDN2_REPO_DIR to the cloned "
            "GatedDeltaNet-2 repository."
        )
    if str(ops_parent) not in sys.path:
        sys.path.insert(0, str(ops_parent))

    from gdn2_ops.chunk_gdn2 import chunk_gdn2  # pylint: disable=import-outside-toplevel

    return chunk_gdn2


def _maybe_compile_linear_rule(fn):
    if os.environ.get("MEGATRON_LINEAR_TORCH_COMPILE", "0") != "1":
        return fn
    if not hasattr(torch, "compile"):
        return fn
    mode = os.environ.get("MEGATRON_LINEAR_TORCH_COMPILE_MODE", "reduce-overhead")
    if os.environ.get("MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS", "0") != "1":
        return torch.compile(
            fn,
            fullgraph=False,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
    return torch.compile(fn, mode=mode, fullgraph=False, dynamic=False)


@dataclass
class GatedDeltaNet2Submodules:
    """Contains the module specs for the input linear, output norm, and output linear layers."""

    in_proj: Union[ModuleSpec, type] = IdentityOp
    out_norm: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


class GatedDeltaNet2(MegatronModule):
    """Gated DeltaNet-2 layer using the external chunk kernel by default."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: GatedDeltaNet2Submodules,
        layer_number: int = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: Optional[float] = None,
        use_qk_l2norm: bool = True,
        A_init_range: Tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config)

        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        self.use_qk_l2norm = use_qk_l2norm
        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        assert pg_collection is not None, "pg_collection must be provided for GatedDeltaNet2"
        self.pg_collection = pg_collection
        self.cp_size = self.pg_collection.cp.size()
        self.tp_size = self.pg_collection.tp.size()
        self.sp_size = self.tp_size if config.sequence_parallel else 1

        self.config = config
        self.hidden_size = config.hidden_size
        self.act_fn = config.activation_func
        self.activation = self.act_fn.__name__
        self.conv_kernel_dim = config.linear_conv_kernel_dim
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads
        self.qk_dim_local_tp = self.qk_dim // self.tp_size
        self.v_dim_local_tp = self.v_dim // self.tp_size
        self.num_key_heads_local_tp = self.num_key_heads // self.tp_size
        self.num_value_heads_local_tp = self.num_value_heads // self.tp_size

        # Faithful GDN-2: the decay (f_proj) and output gate (g_proj) are LOW-RANK 2-layer
        # projections (hidden -> head_v_dim -> key_dim/value_dim), as in the paper/reference, rather
        # than full-rank slices of the fused in_proj. The fused in_proj therefore produces only
        # q, k, v, erase(b), write(w) = 3*qk + 2*v. Scoped to TP=CP=1 (our setup).
        if self.tp_size != 1 or self.cp_size != 1:
            raise NotImplementedError(
                "Faithful low-rank GDN-2 (f_proj/g_proj) is implemented for TP=1 and CP=1."
            )
        self.in_proj_dim = self.qk_dim * 3 + self.v_dim * 2
        if self.config.fp8:
            fp8_align_size = get_fp8_align_size(self.config.fp8_recipe)
            assert self.in_proj_dim % fp8_align_size == 0, (
                "For FP8, the GDN2 input projection output dimension must be aligned."
            )
        self.in_proj = build_module(
            submodules.in_proj,
            self.hidden_size,
            self.in_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )

        # Low-rank decay projection (hidden -> head_v_dim -> key_dim), per GDN-2.
        self.f_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.value_head_dim, bias=False,
                      dtype=config.params_dtype, device=torch.cuda.current_device()),
            nn.Linear(self.value_head_dim, self.qk_dim, bias=False,
                      dtype=config.params_dtype, device=torch.cuda.current_device()),
        )
        # Low-rank output-gate projection (hidden -> head_v_dim -> value_dim, with bias), per GDN-2.
        self.g_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.value_head_dim, bias=False,
                      dtype=config.params_dtype, device=torch.cuda.current_device()),
            nn.Linear(self.value_head_dim, self.v_dim, bias=True,
                      dtype=config.params_dtype, device=torch.cuda.current_device()),
        )

        self.conv_dim = self.qk_dim * 2 + self.v_dim
        self.conv_dim_local_tp = self.conv_dim // self.tp_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim_local_tp,
            out_channels=self.conv_dim_local_tp,
            bias=conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim_local_tp,
            padding=self.conv_kernel_dim - 1,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        setattr(self.conv1d.weight, "tensor_model_parallel", True)
        setattr(self.conv1d.weight, "partition_dim", 0)
        if conv_bias:
            setattr(self.conv1d.bias, "tensor_model_parallel", True)
            setattr(self.conv1d.bias, "partition_dim", 0)

        self.dt_bias = nn.Parameter(
            torch.empty(
                self.qk_dim_local_tp,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.dt_bias, "tensor_model_parallel", True)
        setattr(self.dt_bias, "partition_dim", 0)
        self.A_log = nn.Parameter(
            torch.empty(
                self.num_key_heads_local_tp,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.A_log, "tensor_model_parallel", True)
        setattr(self.A_log, "partition_dim", 0)

        self.use_external_chunk_kernel = (
            os.environ.get("MEGATRON_GDN2_USE_EXTERNAL_CHUNK", "1") == "1"
        )
        if self.use_external_chunk_kernel:
            self.gated_delta_net2_rule = _load_external_chunk_gdn2()
        else:
            self.gated_delta_net2_rule = _maybe_compile_linear_rule(
                torch_recurrent_gated_delta_net2
            )
        self.supports_cler = False
        self.register_parameter("cler_gamma", None)

        self.out_norm = build_module(
            submodules.out_norm,
            config=self.config,
            hidden_size=self.value_head_dim,
            eps=self.config.layernorm_epsilon,
        )

        self.out_proj = build_module(
            submodules.out_proj,
            self.v_dim,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=self.pg_collection.tp,
        )

        self.reset_parameters()

    def reset_parameters(self):
        if self.config.perform_initialization:
            with get_cuda_rng_tracker().fork():
                if self.conv_init is not None:
                    nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
                dt = torch.exp(
                    torch.rand(
                        self.qk_dim_local_tp,
                        dtype=torch.float32,
                        device=torch.cuda.current_device(),
                    )
                    * (math.log(0.1) - math.log(0.001))
                    + math.log(0.001)
                ).clamp(min=1e-4)
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                self.dt_bias.data.copy_(inv_dt.to(dtype=self.config.params_dtype))
                A = torch.empty(
                    self.num_key_heads_local_tp,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                ).uniform_(*self.A_init_range)
                self.A_log.data.copy_(torch.log(A))
                # Low-rank decay/gate projections: init weights with the model init method and
                # zero the output-gate bias (so the gate starts unbiased), per the GDN-2 reference.
                for lin in (*self.f_proj, *self.g_proj):
                    self.config.init_method(lin.weight)
                    if lin.bias is not None:
                        nn.init.zeros_(lin.bias)

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
        del attention_mask, sequence_len_offset, kwargs
        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size * self.cp_size

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "GatedDeltaNet2 does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            raise NotImplementedError("GatedDeltaNet2 does not support inference for now.")

        if packed_seq_params is not None:
            raise NotImplementedError("GatedDeltaNet2 does not support packed sequence for now.")

        nvtx_range_push(suffix="in_proj")
        qkvbw, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        # Low-rank decay and output gate (faithful GDN-2), computed from the hidden states directly
        # (TP=CP=1, so no all-to-all needed). [seq, batch, *] -> [batch, seq, *].
        alpha = self.f_proj(hidden_states).transpose(0, 1)
        gate = self.g_proj(hidden_states).transpose(0, 1)

        qkvbw = tensor_a2a_cp2hp(
            qkvbw,
            seq_dim=0,
            head_dim=-1,
            cp_group=self.pg_collection.cp,
            split_sections=[
                self.qk_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
                self.v_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
                self.qk_dim_local_tp,
            ],
        )

        qkvzbwa = qkvzbwa.transpose(0, 1)
        qkv, gate, erase, write, alpha = torch.split(
            qkvzbwa,
            [
                (self.qk_dim_local_tp * 2 + self.v_dim_local_tp) // self.cp_size,
                self.v_dim_local_tp // self.cp_size,
                self.qk_dim_local_tp // self.cp_size,
                self.v_dim_local_tp // self.cp_size,
                self.qk_dim_local_tp // self.cp_size,
            ],
            dim=-1,
        )
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)

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

        nvtx_range_push(suffix="prepare_qkv_for_gdn2")
        query, key, value, erase, write, alpha = self._prepare_qkv_for_gdn2(
            qkv, erase, write, alpha, batch, seq_len
        )
        nvtx_range_pop(suffix="prepare_qkv_for_gdn2")

        nvtx_range_push(suffix="gdn2_gates")
        A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=self.pg_collection.cp)
        dt_bias_local_cp = get_parameter_local_cp(
            self.dt_bias, dim=0, cp_group=self.pg_collection.cp
        )
        g = self._compute_decay(A_log_local_cp, dt_bias_local_cp, alpha)
        erase = erase.sigmoid()
        write = write.sigmoid()
        nvtx_range_pop(suffix="gdn2_gates")

        nvtx_range_push(suffix="gated_delta_net2_rule")
        if self.use_external_chunk_kernel:
            core_attn_out, _ = self.gated_delta_net2_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                b=erase,
                w=write,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                use_gate_in_kernel=False,
            )
        else:
            core_attn_out, _ = self.gated_delta_net2_rule(
                query,
                key,
                value,
                g=g,
                erase=erase,
                write=write,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
            )
        nvtx_range_pop(suffix="gated_delta_net2_rule")

        nvtx_range_push(suffix="gated_norm")
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix="gated_norm")

        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()
        norm_out = tensor_a2a_hp2cp(
            norm_out, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp
        )

        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")
        return out, out_bias

    @jit_fuser
    def _apply_gated_norm(self, x, gate):
        x_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        y = self.out_norm(x)
        gate = gate.reshape(-1, gate.shape[-1])
        y = y * self.act_fn(gate.float())
        return y.to(x_dtype)

    @jit_fuser
    def _prepare_qkv_for_gdn2(self, qkv, erase, write, alpha, batch, seq_len):
        query_key, value = torch.split(
            qkv,
            [2 * self.qk_dim_local_tp // self.cp_size, self.v_dim_local_tp // self.cp_size],
            dim=-1,
        )
        query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)
        erase = erase.reshape(batch, seq_len, -1, self.key_head_dim)
        write = write.reshape(batch, seq_len, -1, self.value_head_dim)
        alpha = alpha.reshape(batch, seq_len, -1, self.key_head_dim)

        if self.use_qk_l2norm:
            query_key = l2norm(query_key.contiguous())

        split_size = self.qk_dim_local_tp // self.key_head_dim // self.cp_size
        query, key = torch.split(query_key, [split_size, split_size], dim=2)

        if self.num_value_heads // self.num_key_heads > 1:
            repeat_factor = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)
            erase = erase.repeat_interleave(repeat_factor, dim=2)
            alpha = alpha.repeat_interleave(repeat_factor, dim=2)

        return (
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            erase.contiguous(),
            write.contiguous(),
            alpha.contiguous(),
        )

    @jit_fuser
    def _compute_decay(self, A_log_local_cp, dt_bias_local_cp, alpha):
        A = A_log_local_cp.exp().repeat_interleave(self.key_head_dim)
        A = A.view(1, 1, -1, self.key_head_dim)
        dt_bias = dt_bias_local_cp.view(1, 1, -1, self.key_head_dim)
        if self.num_value_heads // self.num_key_heads > 1:
            repeat_factor = self.num_value_heads // self.num_key_heads
            A = A.repeat_interleave(repeat_factor, dim=2)
            dt_bias = dt_bias.repeat_interleave(repeat_factor, dim=2)
        return -A * F.softplus(alpha.float() + dt_bias)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = {}
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={
                "A_log": 0,
                "dt_bias": 0,
            },
            sharded_offsets=sharded_offsets,
            tp_group=(tp_group if tp_group is not None else self.pg_collection.tp),
            dp_cp_group=metadata['dp_cp_group'],
        )

        tp_group = tp_group if tp_group is not None else self.pg_collection.tp
        for name, module in self.named_children():
            if name == "conv1d":
                module_sd = module.state_dict(prefix="", keep_vars=True)
                tp_sharding_map = {"weight": 0}
                if self.conv_bias:
                    tp_sharding_map["bias"] = 0
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f"{prefix}{name}.",
                    tp_sharding_map,
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=metadata['dp_cp_group'],
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
                )
            sharded_state_dict.update(module_sharded_sd)

        in_proj_dim_local_tp = self.in_proj_dim // self.tp_size
        assert sharded_state_dict[f"{prefix}in_proj.weight"].data.size(0) == in_proj_dim_local_tp, (
            in_proj_dim_local_tp,
            sharded_state_dict[f"{prefix}in_proj.weight"],
        )
        sharded_state_dict[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_state_dict[f"{prefix}in_proj.weight"],
            [
                self.qk_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
                self.v_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
                self.qk_dim_local_tp,
            ],
            ["query", "key", "value", "z", "erase", "write", "alpha"],
            0,
        )

        conv_layer_name_list = ["conv1d.weight"]
        if self.conv_bias:
            conv_layer_name_list.append("conv1d.bias")
        for conv_layer_name in conv_layer_name_list:
            sharded_state_dict[f"{prefix}{conv_layer_name}"] = _split_tensor_factory(
                sharded_state_dict[f"{prefix}{conv_layer_name}"],
                [self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp],
                ["query", "key", "value"],
                0,
            )

        return sharded_state_dict

    def backward_dw(self):
        self.in_proj.backward_dw()
        self.out_proj.backward_dw()


def torch_recurrent_gated_delta_net2(
    query,
    key,
    value,
    g,
    erase,
    write,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Token-serial torch implementation of the GDN2 recurrence."""

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    g = g.to(torch.float32)
    erase = erase.to(torch.float32)
    write = write.to(torch.float32)

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (k_head_dim**0.5)
    query = query * scale

    recurrent_state = (
        torch.zeros(
            batch_size,
            num_heads,
            k_head_dim,
            v_head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.empty_like(value)

    for token_idx in range(sequence_length):
        q_t = query[:, token_idx]
        k_t = key[:, token_idx]
        v_t = value[:, token_idx]
        g_t = g[:, token_idx]
        erase_t = erase[:, token_idx]
        write_t = write[:, token_idx]

        recurrent_state = recurrent_state * g_t.exp().unsqueeze(-1)
        erased_key = erase_t * k_t
        erase_read = (recurrent_state * erased_key.unsqueeze(-1)).sum(dim=2)
        v_new = write_t * v_t - erase_read
        recurrent_state = recurrent_state + k_t.unsqueeze(-1) * v_new.unsqueeze(-2)
        core_attn_out[:, token_idx] = (recurrent_state * q_t.unsqueeze(-1)).sum(dim=2)

    if not output_final_state:
        recurrent_state = None
    return core_attn_out.to(initial_dtype), recurrent_state
