# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Helpers for exposing tokenizer operations through HTTP endpoints."""

import inspect
from typing import Any


def _accepts_parameter(func: Any, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _read_attr(obj: Any, *names: str) -> Any:
    """Return the first present attribute value from ``obj``."""
    if obj is None:
        return None
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _detokenize_with_special_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    """Detokenize while preserving special tokens when the tokenizer supports it."""
    if _accepts_parameter(tokenizer.detokenize, "skip_special_tokens"):
        return tokenizer.detokenize(token_ids, skip_special_tokens=False)
    return tokenizer.detokenize(token_ids)


def _special_token_id(tokenizer: Any, *names: str) -> int | None:
    """Return the first usable special-token id from ``tokenizer``."""
    value = _read_attr(tokenizer, *names)
    if isinstance(value, int) and value >= 0:
        return value

    inner = _read_attr(tokenizer, "_tokenizer")
    value = _read_attr(inner, *names)
    if isinstance(value, int) and value >= 0:
        return value

    return None


def _special_token_text(tokenizer: Any, token_name: str, *token_id_names: str) -> str | None:
    """Return a printable special-token string, falling back to detokenization."""
    value = _read_attr(tokenizer, token_name)
    if isinstance(value, str):
        return value

    inner = _read_attr(tokenizer, "_tokenizer")
    value = _read_attr(inner, token_name)
    if isinstance(value, str):
        return value

    token_id = _special_token_id(tokenizer, *token_id_names)
    if token_id is None:
        return None

    try:
        return _detokenize_with_special_tokens(tokenizer, [token_id])
    except Exception:
        return None


def get_tokenizer_info(tokenizer: Any) -> dict[str, Any]:
    """Build the ``lm_eval`` remote-tokenizer metadata payload."""
    return {
        "eos_token": _special_token_text(tokenizer, "eos_token", "eos_id", "eos", "eod"),
        "bos_token": _special_token_text(tokenizer, "bos_token", "bos_id", "bos"),
        "pad_token": _special_token_text(tokenizer, "pad_token", "pad_id", "pad"),
        "chat_template": _read_attr(tokenizer, "chat_template"),
        "vocab_size": _read_attr(tokenizer, "vocab_size"),
    }


def tokenize_for_api(tokenizer: Any, prompt: str, add_special_tokens: bool = False) -> list[int]:
    """Tokenize a prompt for the HTTP API."""
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be a string, got {type(prompt)}")

    token_ids = list(tokenizer.tokenize(prompt))

    if add_special_tokens:
        bos_id = _special_token_id(tokenizer, "bos_id", "bos")
        if bos_id is not None:
            while token_ids and token_ids[0] == bos_id:
                token_ids.pop(0)
            token_ids.insert(0, bos_id)

    return token_ids


def detokenize_for_api(tokenizer: Any, token_ids: list[int]) -> str:
    """Detokenize a token list for the HTTP API."""
    if not isinstance(token_ids, list) or not all(isinstance(t, int) for t in token_ids):
        raise TypeError("tokens must be a list of integers")
    return _detokenize_with_special_tokens(tokenizer, token_ids)
