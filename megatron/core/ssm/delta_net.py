# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
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

from megatron.core.ssm.gated_delta_net import (
    _split_tensor_factory,
    get_parameter_local_cp,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
)
from megatron.core.ssm.cler_utils import (
    chunk_delta_rule_with_residual,
    inject_cler_residual,
    make_cler_gamma_parameter,
)

try:
    from fla.modules.convolution import causal_conv1d
    from fla.modules.l2norm import l2norm
    from fla.ops.delta_rule import chunk_delta_rule

    HAVE_FLA = True
except ImportError:
    causal_conv1d = None
    l2norm = None
    chunk_delta_rule = None

    HAVE_FLA = False

logger = logging.getLogger(__name__)


@dataclass
class DeltaNetSubmodules:
    """Contains the module specs for the input linear, output norm, and output linear layers."""

    in_proj: Union[ModuleSpec, type] = IdentityOp
    out_norm: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


class DeltaNet(MegatronModule):
    """Plain DeltaNet layer with short convolution and chunked delta-rule attention."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DeltaNetSubmodules,
        layer_number: int = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: Optional[float] = None,
        use_qk_l2norm: bool = True,
        pg_collection: ProcessGroupCollection = None,
    ):
        if not HAVE_FLA:
            raise ImportError(
                "FLA is not installed. Please install it with `pip install flash-linear-attention`."
            )

        super().__init__(config)

        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        self.use_qk_l2norm = use_qk_l2norm
        assert pg_collection is not None, "pg_collection must be provided for DeltaNet"
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
        self.num_heads = config.linear_num_key_heads
        self.qk_dim = self.key_head_dim * self.num_heads
        self.v_dim = self.value_head_dim * self.num_heads
        self.qk_dim_local_tp = self.qk_dim // self.tp_size
        self.v_dim_local_tp = self.v_dim // self.tp_size
        self.num_heads_local_tp = self.num_heads // self.tp_size

        # Input projection (hidden_states -> q, k, v, beta)
        self.in_proj_dim = self.qk_dim * 2 + self.v_dim + self.num_heads
        if self.config.fp8:
            fp8_align_size = get_fp8_align_size(self.config.fp8_recipe)
            assert self.in_proj_dim % fp8_align_size == 0, (
                "For FP8, the innermost dimension of the DeltaNet layer "
                "input projection output tensor must be a multiple of 16."
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

        # Conv1d for QKV
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

        self.delta_rule = chunk_delta_rule_with_residual if config.cler_enabled else chunk_delta_rule
        self.supports_cler = True
        self.cler_residual = None
        if self.config.cler_enabled:
            self.cler_gamma = make_cler_gamma_parameter(
                config=self.config,
                num_value_heads_local_tp=self.num_heads_local_tp,
                value_head_dim=self.value_head_dim,
                cp_size=self.cp_size,
                variant_name="DeltaNet",
            )
        else:
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
        # TODO: Deal with attention_mask
        inference_context = deprecate_inference_params(inference_context, inference_params)
        cler_residual = kwargs.pop("cler_residual", None)
        del kwargs
        self.cler_residual = None

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size * self.cp_size

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "DeltaNet does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            raise NotImplementedError("DeltaNet does not support inference for now.")

        if packed_seq_params is not None:
            raise NotImplementedError("DeltaNet does not support packed sequence for now.")

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
        if self.config.deterministic_mode:
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
        else:
            assert self.activation in ["silu", "swish"]
            qkv, _ = causal_conv1d(
                x=qkv,
                weight=conv1d_weight.squeeze(1),
                bias=conv1d_bias,
                activation=self.activation,
                initial_state=None,
                output_final_state=False,
            )
        nvtx_range_pop(suffix="conv1d")

        nvtx_range_push(suffix="prepare_qkv_for_delta_rule")
        query, key, value, beta = self._prepare_qkv_for_delta_rule(qkv, beta, batch, seq_len)
        nvtx_range_pop(suffix="prepare_qkv_for_delta_rule")

        if self.config.cler_enabled:
            value = inject_cler_residual(
                value=value,
                cler_residual=cler_residual,
                cler_gamma=self.cler_gamma,
                config=self.config,
                variant_name="DeltaNet",
            )

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

    def _apply_norm(self, x):
        x_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        y = self.out_norm(x)
        return y.to(x_dtype)

    def _prepare_qkv_for_delta_rule(self, qkv, beta, batch, seq_len):
        query_key, value = torch.split(
            qkv,
            [2 * self.qk_dim_local_tp // self.cp_size, self.v_dim_local_tp // self.cp_size],
            dim=-1,
        )
        query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)

        if self.use_qk_l2norm:
            query_key = l2norm(query_key.contiguous())

        split_size = self.qk_dim_local_tp // self.key_head_dim // self.cp_size
        query, key = torch.split(query_key, [split_size, split_size], dim=2)

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        beta = beta.contiguous()
        return query, key, value, beta

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = {}
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        tensor_parallel_layers_axis_map = {}
        if self.config.cler_enabled and self.config.cler_gamma_mode in {"head", "channel"}:
            tensor_parallel_layers_axis_map["cler_gamma"] = 0
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map=tensor_parallel_layers_axis_map,
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
                self.num_heads_local_tp,
            ],
            ["query", "key", "value", "beta"],
            0,
        )

        conv_layer_name_list = ["conv1d.weight"]
        assert (
            sharded_state_dict[f"{prefix}conv1d.weight"].data.size(0) == self.conv_dim_local_tp
        ), (self.conv_dim_local_tp, sharded_state_dict[f"{prefix}conv1d.weight"])
        if self.conv_bias:
            conv_layer_name_list.append("conv1d.bias")
            assert (
                sharded_state_dict[f"{prefix}conv1d.bias"].data.size(0) == self.conv_dim_local_tp
            ), (self.conv_dim_local_tp, sharded_state_dict[f"{prefix}conv1d.bias"])

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
