import pathlib
import sys

import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PACKAGE_DIR = _REPO_ROOT / "_research" / "packages"
for _path in (_REPO_ROOT, _PACKAGE_DIR):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from megatron.core.ssm.cler_delta_net_pytorch import torch_chunk_cler_delta_rule
from megatron.core.ssm.cler_utils import (
    HAVE_FLA,
    chunk_delta_rule_with_residual,
    chunk_gated_delta_rule_with_residual,
)
from megatron.core.ssm.gated_delta_net_pytorch import torch_chunk_gated_delta_rule


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available."),
    pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed."),
]

_BF16_FAST_ATOL = 2e-1
_BF16_FAST_RTOL = 5e-2


def _make_delta_inputs(seed=123, requires_grad=False):
    torch.manual_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    batch, seq_len, heads, key_dim, value_dim = 1, 8, 2, 16, 16
    query = torch.randn(batch, seq_len, heads, key_dim, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn(batch, seq_len, heads, value_dim, device=device, dtype=dtype)
    beta = torch.sigmoid(
        torch.randn(batch, seq_len, heads, device=device, dtype=torch.float32)
    ).to(dtype)
    if requires_grad:
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)
        beta.requires_grad_(True)
    return query, key, value, beta


def _make_gated_inputs(seed=123, requires_grad=False):
    query, key, value, beta = _make_delta_inputs(seed=seed, requires_grad=requires_grad)
    g = torch.zeros_like(beta)
    if requires_grad:
        g.requires_grad_(True)
    return query, key, value, g, beta


def test_fast_delta_rule_return_residual_matches_pytorch_small_tensor():
    query, key, value, beta = _make_delta_inputs()

    fast_output, fast_state, fast_residual = chunk_delta_rule_with_residual(
        query, key, value, beta=beta, return_residual=True
    )
    ref_output, ref_state, ref_residual = torch_chunk_cler_delta_rule(
        query, key, value, beta=beta, return_residual=True
    )

    assert fast_state is None
    assert ref_state is None
    assert fast_output.shape == ref_output.shape == value.shape
    assert fast_residual.shape == ref_residual.shape == value.shape
    torch.testing.assert_close(
        fast_output, ref_output, atol=_BF16_FAST_ATOL, rtol=_BF16_FAST_RTOL
    )
    torch.testing.assert_close(
        fast_residual, ref_residual, atol=_BF16_FAST_ATOL, rtol=_BF16_FAST_RTOL
    )


def test_fast_gated_delta_rule_return_residual_matches_pytorch_small_tensor():
    query, key, value, g, beta = _make_gated_inputs()

    fast_output, fast_state, fast_residual = chunk_gated_delta_rule_with_residual(
        query, key, value, g=g, beta=beta, return_residual=True
    )
    ref_output, ref_state, ref_residual = torch_chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, return_residual=True
    )

    assert fast_state is None
    assert ref_state is None
    assert fast_output.shape == ref_output.shape == value.shape
    assert fast_residual.shape == ref_residual.shape == value.shape
    torch.testing.assert_close(
        fast_output, ref_output, atol=_BF16_FAST_ATOL, rtol=_BF16_FAST_RTOL
    )
    torch.testing.assert_close(
        fast_residual, ref_residual, atol=_BF16_FAST_ATOL, rtol=_BF16_FAST_RTOL
    )


@pytest.mark.parametrize("rule", ["delta", "gated_delta"])
def test_fast_rule_backward_includes_residual_gradient(rule):
    if rule == "delta":
        query, key, value, beta = _make_delta_inputs(seed=321, requires_grad=True)
        output, _, residual = chunk_delta_rule_with_residual(
            query, key, value, beta=beta, return_residual=True
        )
    else:
        query, key, value, g, beta = _make_gated_inputs(seed=321, requires_grad=True)
        output, _, residual = chunk_gated_delta_rule_with_residual(
            query, key, value, g=g, beta=beta, return_residual=True
        )
    output.float().square().mean().backward()
    value_grad_without_residual = value.grad.detach().clone()

    if rule == "delta":
        query, key, value, beta = _make_delta_inputs(seed=321, requires_grad=True)
        output, _, residual = chunk_delta_rule_with_residual(
            query, key, value, beta=beta, return_residual=True
        )
    else:
        query, key, value, g, beta = _make_gated_inputs(seed=321, requires_grad=True)
        output, _, residual = chunk_gated_delta_rule_with_residual(
            query, key, value, g=g, beta=beta, return_residual=True
        )
    (output.float().square().mean() + residual.float().square().mean()).backward()
    value_grad_with_residual = value.grad.detach()

    assert not torch.allclose(value_grad_without_residual, value_grad_with_residual)
