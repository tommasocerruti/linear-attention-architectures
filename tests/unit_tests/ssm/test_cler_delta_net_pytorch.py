import pathlib
import sys

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_linear_attention_pattern,
    is_linear_attention_variant,
)
from megatron.core.ssm.cler_delta_net_pytorch import torch_chunk_cler_delta_rule
from megatron.core.ssm.delta_net_pytorch import torch_chunk_delta_rule
from megatron.core.transformer import TransformerConfig


def _make_rule_inputs(seq_len=1, batch=1, heads=2, key_dim=3, value_dim=5):
    query = torch.ones(batch, seq_len, heads, key_dim)
    key = torch.ones_like(query)
    value = torch.arange(
        1, batch * seq_len * heads * value_dim + 1, dtype=torch.float32
    ).view(batch, seq_len, heads, value_dim)
    beta = torch.ones(batch, seq_len, heads)
    return query, key, value, beta


def _make_linear_config(**overrides):
    kwargs = dict(
        num_layers=6,
        hidden_size=64,
        num_attention_heads=4,
        experimental_attention_variant="cler_delta_net_pytorch",
        linear_attention_freq=3,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
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


def test_cler_delta_rule_default_matches_delta_rule():
    query, key, value, beta = _make_rule_inputs(seq_len=3)

    cler_output, cler_final_state = torch_chunk_cler_delta_rule(
        query, key, value, beta, chunk_size=2
    )
    delta_output, delta_final_state = torch_chunk_delta_rule(
        query, key, value, beta, chunk_size=2
    )

    assert cler_final_state is None
    assert delta_final_state is None
    assert torch.allclose(cler_output, delta_output)


def test_cler_delta_rule_returns_residual_shape_and_dtype():
    query, key, value, beta = _make_rule_inputs(seq_len=3)

    output, final_state, residual = torch_chunk_cler_delta_rule(
        query, key, value, beta, chunk_size=2, return_residual=True
    )

    assert output.shape == value.shape
    assert final_state is None
    assert residual.shape == value.shape
    assert residual.dtype == value.dtype


def test_cler_delta_rule_single_token_residual_equals_value():
    query, key, value, beta = _make_rule_inputs(seq_len=1)

    _, _, residual = torch_chunk_cler_delta_rule(
        query, key, value, beta, chunk_size=1, return_residual=True
    )

    assert torch.allclose(residual, value)


def test_cler_delta_net_pytorch_uses_linear_attention_pattern():
    config = _make_linear_config()

    assert is_linear_attention_variant("cler_delta_net_pytorch")
    assert get_linear_attention_pattern(config) == [1, 1, 0, 1, 1, 0]


def test_cler_accepts_pytorch_delta_net_variant():
    config = _make_linear_config(cler_enabled=True, cler_gamma_init=0.125)

    assert config.cler_enabled
    assert config.cler_gamma_init == 0.125


def test_cler_delta_accepts_head_gamma_and_residual_norm_config():
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


def test_cler_accepts_fast_delta_net_variant():
    config = _make_linear_config(
        experimental_attention_variant="delta_net",
        cler_enabled=True,
        cler_gamma_init=0.125,
    )

    assert config.cler_enabled
    assert config.experimental_attention_variant == "delta_net"
    assert config.cler_gamma_init == 0.125


def test_cler_still_rejects_plain_delta_net_pytorch_variant():
    _assert_raises(
        ValueError,
        _make_linear_config,
        experimental_attention_variant="delta_net_pytorch",
        cler_enabled=True,
        contains="cler_delta_net_pytorch",
    )


def _run_directly():
    for name, test_fn in sorted(globals().items()):
        if name.startswith("test_") and callable(test_fn):
            test_fn()
            print(f"{name}: ok")


if __name__ == "__main__":
    _run_directly()
