from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import _split_tensor_factory
from megatron.core.ssm.mamba_context_parallel import (
    _redo_attention_load_balancing,
    _undo_attention_load_balancing,
)
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
from megatron.core.utils import deprecate_inference_params

try:
    from fla.modules import FusedRMSNormGated
    from fla.modules.convolution import causal_conv1d
    from fla.ops.cp import build_cp_context
    from fla.ops.kda import chunk_kda

    HAVE_FLA = True
except ImportError:
    FusedRMSNormGated = None
    causal_conv1d = None
    build_cp_context = None
    chunk_kda = None
    HAVE_FLA = False


@dataclass
class KimiDeltaAttentionSubmodules:
    """Submodule builders for Megatron-native KDA.

    This follows Megatron's ModuleSpec pattern so the backend decides which TP linear
    implementations are instantiated, instead of hard-coding them inside the module.
    """

    in_proj: Union[ModuleSpec, type] = IdentityOp
    f_out_proj: Union[ModuleSpec, type] = IdentityOp
    g_out_proj: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


class KimiDeltaAttention(MegatronModule):
    """Megatron-native KDA wrapper using TP-aware projections and FLA kernels."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: KimiDeltaAttentionSubmodules,
        layer_number: int = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: Optional[float] = None,
        A_init_range: tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
    ):
        if not HAVE_FLA:
            raise ImportError(
                "FLA is not installed. Please install it with `pip install flash-linear-attention`."
            )

        super().__init__(config)

        if config.sequence_parallel:
            raise ValueError(
                "KDA multi-GPU training supports DP/TP/CP in this milestone, but "
                "--sequence-parallel is not yet supported. Disable --sequence-parallel "
                "for --experimental-attention-variant kda."
            )
        if config.deterministic_mode:
            raise NotImplementedError("KDA does not support deterministic_mode in this wrapper.")

        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        self.A_init_range = A_init_range
        self.config = config

        assert pg_collection is not None, "pg_collection must be provided for KDA"
        self.pg_collection = pg_collection
        self.cp_size = self.pg_collection.cp.size()
        self.tp_size = self.pg_collection.tp.size()

        self.hidden_size = config.hidden_size
        self.kda_use_flashkda = config.kda_use_flashkda
        self.conv_kernel_dim = config.linear_conv_kernel_dim
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads

        assert self.key_head_dim is not None
        assert self.value_head_dim is not None
        assert self.num_key_heads is not None
        assert self.num_value_heads is not None

        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads
        self.gate_dim = self.key_head_dim * self.num_value_heads

        self.num_key_heads_local_tp = self.num_key_heads // self.tp_size
        self.num_value_heads_local_tp = self.num_value_heads // self.tp_size
        self.qk_dim_local_tp = self.qk_dim // self.tp_size
        self.v_dim_local_tp = self.v_dim // self.tp_size
        self.gate_dim_local_tp = self.gate_dim // self.tp_size

        self.in_proj_dim = self.qk_dim * 2 + self.v_dim + self.num_value_heads
        if self.config.fp8:
            fp8_align_size = get_fp8_align_size(self.config.fp8_recipe)
            assert self.in_proj_dim % fp8_align_size == 0, (
                "For FP8, the KDA input projection output tensor must be a multiple of 16."
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

        bottleneck_device = None if self.config.use_cpu_initialization else torch.cuda.current_device()
        self.bottleneck_proj = nn.Linear(
            self.hidden_size,
            2 * self.value_head_dim,
            bias=False,
            device=bottleneck_device,
            dtype=self.config.params_dtype,
        )

        self.f_out_proj = build_module(
            submodules.f_out_proj,
            self.value_head_dim,
            self.gate_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kda_f",
            tp_group=self.pg_collection.tp,
        )
        self.g_out_proj = build_module(
            submodules.g_out_proj,
            self.value_head_dim,
            self.v_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kda_g",
            tp_group=self.pg_collection.tp,
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

        self.A_log = nn.Parameter(
            torch.empty(
                self.num_value_heads_local_tp,
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.A_log, "tensor_model_parallel", True)
        setattr(self.A_log, "partition_dim", 0)
        setattr(self.A_log, "_no_weight_decay", True)

        self.dt_bias = nn.Parameter(
            torch.empty(
                self.gate_dim_local_tp,
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.dt_bias, "tensor_model_parallel", True)
        setattr(self.dt_bias, "partition_dim", 0)
        setattr(self.dt_bias, "_no_weight_decay", True)

        self.out_norm = FusedRMSNormGated(
            self.value_head_dim,
            activation="sigmoid",
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
        if not self.config.perform_initialization:
            return

        with get_cuda_rng_tracker().fork():
            self.config.init_method(self.bottleneck_proj.weight)
            if self.conv_init is not None:
                nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
            if self.conv_bias:
                nn.init.zeros_(self.conv1d.bias)

            A = torch.empty(
                self.num_value_heads_local_tp,
                dtype=torch.float32,
                device=self.A_log.device,
            ).uniform_(*self.A_init_range)
            self.A_log.data.copy_(torch.log(A))

            dt = torch.exp(
                torch.rand(
                    self.gate_dim_local_tp,
                    dtype=torch.float32,
                    device=self.dt_bias.device,
                )
                * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            ).clamp_(min=1e-4)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias.data.copy_(inv_dt)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context is not None:
            raise NotImplementedError("KDA does not support inference cache integration yet.")
        if packed_seq_params is not None:
            raise NotImplementedError("KDA does not support packed sequences yet.")
        if sequence_len_offset is not None:
            raise NotImplementedError("KDA does not support CUDA-graph sequence offsets yet.")

        if self.cp_size > 1:
            hidden_states = _undo_attention_load_balancing(hidden_states, self.cp_size)

        seq_len_local, batch, _ = hidden_states.shape
        seq_len_global = seq_len_local * self.cp_size
        local_token_count = batch * seq_len_local

        qkvbeta, _ = self.in_proj(hidden_states)
        bottleneck = self.bottleneck_proj(hidden_states)
        f_hidden, g_hidden = torch.chunk(bottleneck, 2, dim=-1)
        gate_logits, _ = self.f_out_proj(f_hidden)
        gate_branch, _ = self.g_out_proj(g_hidden)

        qkv, beta = torch.split(
            qkvbeta,
            [2 * self.qk_dim_local_tp + self.v_dim_local_tp, self.num_value_heads_local_tp],
            dim=-1,
        )

        qkv = self._flatten_batch_time(qkv)
        beta = self._flatten_batch_time(beta)
        gate_logits = self._flatten_batch_time(gate_logits)
        gate_branch = self._flatten_batch_time(gate_branch)

        cu_seqlens = self._build_equal_length_cu_seqlens(
            seq_len_global, batch, hidden_states.device
        )
        cp_context = (
            build_cp_context(
                cu_seqlens,
                group=self.pg_collection.cp,
                conv1d_kernel_size=self.conv_kernel_dim,
            )
            if self.cp_size > 1
            else None
        )

        qkv, _ = causal_conv1d(
            x=qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation="silu",
            cp_context=cp_context,
            cu_seqlens=None if cp_context is not None else cu_seqlens,
        )

        q, k, v = torch.split(
            qkv,
            [self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp],
            dim=-1,
        )

        q = q.reshape(1, local_token_count, self.num_key_heads_local_tp, self.key_head_dim).contiguous()
        k = k.reshape(1, local_token_count, self.num_key_heads_local_tp, self.key_head_dim).contiguous()
        v = v.reshape(
            1, local_token_count, self.num_value_heads_local_tp, self.value_head_dim
        ).contiguous()
        gate_logits = gate_logits.reshape(
            1, local_token_count, self.num_value_heads_local_tp, self.key_head_dim
        ).contiguous()
        beta = beta.reshape(1, local_token_count, self.num_value_heads_local_tp).sigmoid().contiguous()
        gate_branch = gate_branch.reshape(
            1, local_token_count, self.num_value_heads_local_tp, self.value_head_dim
        ).contiguous()

        with self._flashkda_dispatch():
            core_attn_out, _ = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=gate_logits,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=False,
                cu_seqlens=None if cp_context is not None else cu_seqlens,
                safe_gate=False,
                lower_bound=None,
                disable_recompute=False,
                cp_context=cp_context,
            )

        norm_out = self.out_norm(core_attn_out, gate_branch)
        norm_out = norm_out.reshape(batch, seq_len_local, self.v_dim_local_tp)
        norm_out = norm_out.transpose(0, 1).contiguous()

        out, out_bias = self.out_proj(norm_out)

        if self.cp_size > 1:
            out = _redo_attention_load_balancing(out, self.cp_size)

        return out, out_bias

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = {}
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={"A_log": 0, "dt_bias": 0},
            sharded_offsets=sharded_offsets,
            tp_group=(tp_group if tp_group is not None else self.pg_collection.tp),
            dp_cp_group=metadata["dp_cp_group"],
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
                    dp_cp_group=metadata["dp_cp_group"],
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
                )
            sharded_state_dict.update(module_sharded_sd)

        sharded_state_dict[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_state_dict[f"{prefix}in_proj.weight"],
            [
                self.qk_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
                self.num_value_heads_local_tp,
            ],
            ["query", "key", "value", "beta"],
            0,
        )
        if f"{prefix}in_proj.bias" in sharded_state_dict:
            sharded_state_dict[f"{prefix}in_proj.bias"] = _split_tensor_factory(
                sharded_state_dict[f"{prefix}in_proj.bias"],
                [
                    self.qk_dim_local_tp,
                    self.qk_dim_local_tp,
                    self.v_dim_local_tp,
                    self.num_value_heads_local_tp,
                ],
                ["query", "key", "value", "beta"],
                0,
            )

        conv_layer_names = ["conv1d.weight"]
        if self.conv_bias:
            conv_layer_names.append("conv1d.bias")
        for conv_layer_name in conv_layer_names:
            sharded_state_dict[f"{prefix}{conv_layer_name}"] = _split_tensor_factory(
                sharded_state_dict[f"{prefix}{conv_layer_name}"],
                [self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp],
                ["query", "key", "value"],
                0,
            )

        return sharded_state_dict

    def backward_dw(self):
        self.out_proj.backward_dw()
        self.g_out_proj.backward_dw()
        self.f_out_proj.backward_dw()
        self.in_proj.backward_dw()

    @contextmanager
    def _flashkda_dispatch(self):
        env_name = "FLA_FLASH_KDA"
        previous = os.environ.get(env_name)
        os.environ[env_name] = "1" if self.kda_use_flashkda else "0"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = previous

    @staticmethod
    def _flatten_batch_time(tensor: Tensor) -> Tensor:
        batch, seq_len, hidden = tensor.transpose(0, 1).shape
        return tensor.transpose(0, 1).contiguous().reshape(1, batch * seq_len, hidden)

    @staticmethod
    def _build_equal_length_cu_seqlens(
        seq_len_global: int, batch: int, device: torch.device
    ) -> Tensor:
        return torch.arange(
            0,
            (batch + 1) * seq_len_global,
            seq_len_global,
            device=device,
            dtype=torch.long,
        )
