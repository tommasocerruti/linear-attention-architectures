from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import deprecate_inference_params

try:
    from fla.layers.kda import KimiDeltaAttention as FLAKimiDeltaAttention

    HAVE_FLA = True
except ImportError:
    FLAKimiDeltaAttention = None
    HAVE_FLA = False


class KimiDeltaAttention(MegatronModule):
    """Megatron wrapper around FLA's Kimi Delta Attention layer.

    This first-pass integration is intentionally scoped to the repo's bring-up path:
    single-GPU training with no TP/CP/SP, no packed sequences, and no inference cache.
    """

    def __init__(
        self,
        config,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        submodules=None,
        **kwargs,
    ):
        if not HAVE_FLA:
            raise ImportError(
                "FLA is not installed. Please install it with `pip install flash-linear-attention`."
            )

        super().__init__(config=config)
        self.layer_number = layer_number
        self.pg_collection = pg_collection

        if config.tensor_model_parallel_size != 1:
            raise NotImplementedError("KDA currently only supports tensor_model_parallel_size=1.")
        if config.context_parallel_size != 1:
            raise NotImplementedError("KDA currently only supports context_parallel_size=1.")
        if config.sequence_parallel:
            raise NotImplementedError("KDA currently does not support sequence parallelism.")
        if config.linear_attention_freq is None:
            raise ValueError("KDA requires --linear-attention-freq to define the hybrid pattern.")

        head_dim = config.linear_key_head_dim
        value_dim = config.linear_value_head_dim
        if head_dim is None or value_dim is None:
            raise ValueError("KDA requires linear_key_head_dim and linear_value_head_dim.")

        expand_v = value_dim / head_dim
        if not math.isclose(round(expand_v) * head_dim, value_dim, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "KDA requires linear_value_head_dim to be an integer multiple of "
                "linear_key_head_dim."
            )

        self.kda = FLAKimiDeltaAttention(
            hidden_size=config.hidden_size,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=config.linear_num_key_heads,
            num_v_heads=config.linear_num_value_heads,
            mode="chunk",
            use_short_conv=True,
            conv_size=config.linear_conv_kernel_dim,
            conv_bias=False,
            layer_idx=layer_number,
            norm_eps=config.layernorm_epsilon,
        )

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

        # FLA expects [batch, seq, hidden]. Standard Megatron pretraining uses a causal mask
        # and no padding mask, so we drop the 4D causal attention mask here.
        fla_attention_mask = attention_mask if attention_mask is not None and attention_mask.dim() == 2 else None

        hidden_states = hidden_states.transpose(0, 1).contiguous()
        output, _, _ = self.kda(
            hidden_states=hidden_states,
            attention_mask=fla_attention_mask,
            use_cache=False,
        )
        output = output.transpose(0, 1).contiguous()
        return output, None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        return sharded_state_dict_default(self, prefix, sharded_offsets, metadata, tp_group)

    def backward_dw(self):
        for module in self.modules():
            if module is self:
                continue
            backward_dw = getattr(module, "backward_dw", None)
            if callable(backward_dw):
                backward_dw()
