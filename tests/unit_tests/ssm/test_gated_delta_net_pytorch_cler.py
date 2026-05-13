import pathlib
import sys
from types import SimpleNamespace

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_linear_attention_pattern,
    is_linear_attention_variant,
)
from megatron.core.ssm.gated_delta_net_pytorch import torch_chunk_gated_delta_rule
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock


def _make_rule_inputs(seq_len=1, batch=1, heads=2, head_dim=3):
    query = torch.ones(batch, seq_len, heads, head_dim)
    key = torch.ones_like(query)
    value = torch.arange(
        1, batch * seq_len * heads * head_dim + 1, dtype=torch.float32
    ).view(batch, seq_len, heads, head_dim)
    g = torch.zeros(batch, seq_len, heads)
    beta = torch.ones(batch, seq_len, heads)
    return query, key, value, g, beta


def _make_linear_config(**overrides):
    kwargs = dict(
        num_layers=6,
        hidden_size=64,
        num_attention_heads=4,
        experimental_attention_variant="gated_delta_net_pytorch",
        linear_attention_freq=3,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _assert_raises(expected_exception, fn, *args, contains=None, **kwargs):
    try:
        fn(*args, **kwargs)
    except expected_exception as error:
        if contains is not None:
            assert contains in str(error)
        return
    raise AssertionError(f"Expected {expected_exception.__name__} to be raised")


def test_torch_chunk_gated_delta_rule_default_returns_two_outputs():
    query, key, value, g, beta = _make_rule_inputs(seq_len=2)

    outputs = torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=1)

    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    output, final_state = outputs
    assert output.shape == value.shape
    assert final_state is None


def test_torch_chunk_gated_delta_rule_returns_cler_residual_shape_and_dtype():
    query, key, value, g, beta = _make_rule_inputs(seq_len=3)

    output, final_state, residual = torch_chunk_gated_delta_rule(
        query, key, value, g, beta, chunk_size=2, return_residual=True
    )

    assert output.shape == value.shape
    assert final_state is None
    assert residual.shape == value.shape
    assert residual.dtype == value.dtype


def test_torch_chunk_gated_delta_rule_single_token_residual_equals_value():
    query, key, value, g, beta = _make_rule_inputs(seq_len=1)

    _, _, residual = torch_chunk_gated_delta_rule(
        query, key, value, g, beta, chunk_size=1, return_residual=True
    )

    assert torch.allclose(residual, value)


def test_cler_value_injection_gamma_zero_and_one():
    query, key, value, g, beta = _make_rule_inputs(seq_len=2)
    previous_residual = torch.full_like(value, 0.25)

    baseline, _ = torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=1)
    gamma_zero, _ = torch_chunk_gated_delta_rule(
        query, key, value + 0.0 * previous_residual, g, beta, chunk_size=1
    )
    gamma_one, _ = torch_chunk_gated_delta_rule(
        query, key, value + previous_residual, g, beta, chunk_size=1
    )

    assert torch.allclose(gamma_zero, baseline)
    assert not torch.allclose(gamma_one, baseline)


def test_gated_delta_net_pytorch_uses_linear_attention_pattern():
    config = _make_linear_config()

    assert is_linear_attention_variant("gated_delta_net_pytorch")
    assert get_linear_attention_pattern(config) == [1, 1, 0, 1, 1, 0]


def test_cler_accepts_pytorch_gated_delta_net_variant():
    config = _make_linear_config(cler_enabled=True, cler_gamma_init=0.125)

    assert config.cler_enabled
    assert config.cler_gamma_init == 0.125


def test_cler_accepts_head_gamma_and_residual_norm_config():
    config = _make_linear_config(
        cler_enabled=True,
        cler_gamma_init=0.01,
        cler_gamma_mode="head",
        cler_normalize_residual=True,
        cler_residual_norm_eps=1e-5,
    )

    assert config.cler_gamma_mode == "head"
    assert config.cler_normalize_residual
    assert config.cler_residual_norm_eps == 1e-5


def test_cler_rejects_invalid_gamma_mode():
    _assert_raises(
        ValueError,
        _make_linear_config,
        cler_enabled=True,
        cler_gamma_mode="channel",
        contains="cler_gamma_mode",
    )


def test_cler_rejects_nonpositive_residual_norm_eps():
    _assert_raises(
        ValueError,
        _make_linear_config,
        cler_enabled=True,
        cler_normalize_residual=True,
        cler_residual_norm_eps=0.0,
        contains="cler_residual_norm_eps",
    )


def test_cler_rejects_non_pytorch_gated_delta_net_variant():
    _assert_raises(
        ValueError,
        _make_linear_config,
        experimental_attention_variant="gated_delta_net",
        cler_enabled=True,
        contains="gated_delta_net_pytorch",
    )


def test_cler_residual_carries_across_non_cler_layers():
    block = SimpleNamespace(config=SimpleNamespace(cler_enabled=True))
    residual = torch.tensor([1.0])
    non_cler_layer = SimpleNamespace(supports_cler=False)

    next_residual = TransformerBlock._get_next_cler_residual(
        block, non_cler_layer, residual
    )

    assert next_residual is residual


def test_cler_layer_replaces_carried_residual():
    block = SimpleNamespace(config=SimpleNamespace(cler_enabled=True))
    previous_residual = torch.tensor([1.0])
    produced_residual = torch.tensor([2.0])
    cler_layer = SimpleNamespace(supports_cler=True, cler_residual=produced_residual)

    next_residual = TransformerBlock._get_next_cler_residual(
        block, cler_layer, previous_residual
    )

    assert next_residual is produced_residual


def test_cler_residual_is_cleared_when_cler_disabled():
    block = SimpleNamespace(config=SimpleNamespace(cler_enabled=False))
    residual = torch.tensor([1.0])
    non_cler_layer = SimpleNamespace(supports_cler=False)

    assert TransformerBlock._get_next_cler_residual(block, non_cler_layer, residual) is None


def _run_directly():
    for name, test_fn in sorted(globals().items()):
        if name.startswith("test_") and callable(test_fn):
            test_fn()
            print(f"{name}: ok")


if __name__ == "__main__":
    _run_directly()
