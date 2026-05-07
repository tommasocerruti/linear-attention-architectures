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
from megatron.core.ssm.linear_transformer_pytorch import torch_chunk_linear_transformer_rule
from megatron.core.transformer import TransformerConfig


def _naive_additive_linear_rule(query, key, value):
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    scale = 1 / (query.shape[-1] ** 0.5)
    state = torch.zeros(
        query.shape[0],
        query.shape[2],
        query.shape[3],
        value.shape[3],
        dtype=torch.float32,
        device=query.device,
    )
    outputs = []
    for i in range(query.shape[1]):
        state = state + key[:, i].unsqueeze(-1) @ value[:, i].unsqueeze(-2)
        outputs.append(((query[:, i] * scale).unsqueeze(-2) @ state).squeeze(-2))
    return torch.stack(outputs, dim=1).to(value.dtype)


def _make_linear_config(**overrides):
    kwargs = dict(
        num_layers=6,
        hidden_size=64,
        num_attention_heads=4,
        experimental_attention_variant="linear_transformer_pytorch",
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


def test_torch_chunk_linear_transformer_rule_matches_naive_recurrence():
    torch.manual_seed(123)
    query = torch.randn(2, 5, 3, 4)
    key = torch.randn(2, 5, 3, 4)
    value = torch.randn(2, 5, 3, 6)

    output, final_state = torch_chunk_linear_transformer_rule(
        query, key, value, chunk_size=2, output_final_state=False
    )

    expected = _naive_additive_linear_rule(query, key, value)
    assert final_state is None
    assert output.shape == value.shape
    assert torch.allclose(output, expected, atol=1e-5, rtol=1e-5)


def test_torch_chunk_linear_transformer_rule_returns_final_state():
    query = torch.ones(1, 3, 2, 4)
    key = torch.ones_like(query)
    value = torch.ones(1, 3, 2, 5)

    _, final_state = torch_chunk_linear_transformer_rule(
        query, key, value, chunk_size=2, output_final_state=True
    )

    assert final_state.shape == (1, 2, 4, 5)
    assert torch.allclose(final_state, torch.full_like(final_state, 3.0))


def test_linear_transformer_pytorch_uses_linear_attention_pattern():
    config = _make_linear_config()

    assert is_linear_attention_variant("linear_transformer_pytorch")
    assert get_linear_attention_pattern(config) == [1, 1, 0, 1, 1, 0]


def test_linear_transformer_rejects_mismatched_value_heads():
    _assert_raises(
        AssertionError,
        _make_linear_config,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        contains="matching q/k and value head counts",
    )


def _run_directly():
    for name, test_fn in sorted(globals().items()):
        if name.startswith("test_") and callable(test_fn):
            test_fn()
            print(f"{name}: ok")


if __name__ == "__main__":
    _run_directly()
