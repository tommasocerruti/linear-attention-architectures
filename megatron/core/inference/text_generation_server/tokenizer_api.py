# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Helpers for exposing tokenizer operations through HTTP endpoints."""

import inspect
from typing import Any, Dict, List, Optional


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


def _detokenize_with_special_tokens(tokenizer: Any, token_ids: List[int]) -> str:
    """Detokenize while preserving special tokens when the tokenizer supports it."""
    if _accepts_parameter(tokenizer.detokenize, "skip_special_tokens"):
        return tokenizer.detokenize(token_ids, skip_special_tokens=False)
    return tokenizer.detokenize(token_ids)


def _special_token_id(tokenizer: Any, *names: str) -> Optional[int]:
    """Return the first usable special-token id from ``tokenizer``."""
    value = _read_attr(tokenizer, *names)
    if isinstance(value, int) and value >= 0:
        return value

    inner = _read_attr(tokenizer, "_tokenizer")
    value = _read_attr(inner, *names)
    if isinstance(value, int) and value >= 0:
        return value

    return None


def _special_token_piece(tokenizer: Any, token_id: int) -> Optional[str]:
    """Return the tokenizer piece for a special id when available."""
    for candidate in (tokenizer, _read_attr(tokenizer, "_tokenizer")):
        ids_to_tokens = _read_attr(candidate, "ids_to_tokens")
        if callable(ids_to_tokens):
            try:
                value = ids_to_tokens([token_id])[0]
            except Exception:
                value = None
            if isinstance(value, str) and value:
                return value

        sentencepiece_tokenizer = _read_attr(candidate, "tokenizer")
        id_to_piece = _read_attr(sentencepiece_tokenizer, "id_to_piece")
        if callable(id_to_piece):
            try:
                value = id_to_piece(token_id)
            except Exception:
                value = None
            if isinstance(value, str) and value:
                return value

    return None


def _special_token_text(
    tokenizer: Any, token_name: str, *token_id_names: str
) -> Optional[str]:
    """Return a printable special-token string, falling back to detokenization."""
    value = _read_attr(tokenizer, token_name)
    if isinstance(value, str) and value:
        return value

    inner = _read_attr(tokenizer, "_tokenizer")
    value = _read_attr(inner, token_name)
    if isinstance(value, str) and value:
        return value

    token_id = _special_token_id(tokenizer, *token_id_names)
    if token_id is None:
        return None

    value = _special_token_piece(tokenizer, token_id)
    if value:
        return value

    try:
        value = _detokenize_with_special_tokens(tokenizer, [token_id])
    except Exception:
        return None
    return value or None


def get_tokenizer_info(tokenizer: Any) -> Dict[str, Any]:
    """Build the ``lm_eval`` remote-tokenizer metadata payload."""
    return {
        "eos_token": _special_token_text(tokenizer, "eos_token", "eos_id", "eos", "eod"),
        "bos_token": _special_token_text(tokenizer, "bos_token", "bos_id", "bos"),
        "pad_token": _special_token_text(tokenizer, "pad_token", "pad_id", "pad"),
        "chat_template": _read_attr(tokenizer, "chat_template"),
        "vocab_size": _read_attr(tokenizer, "vocab_size"),
    }


def tokenize_for_api(tokenizer: Any, prompt: str, add_special_tokens: bool = False) -> List[int]:
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


def detokenize_for_api(tokenizer: Any, token_ids: List[int]) -> str:
    """Detokenize a token list for the HTTP API."""
    if not isinstance(token_ids, list) or not all(isinstance(t, int) for t in token_ids):
        raise TypeError("tokens must be a list of integers")

    special_token_ids = {
        token_id
        for token_id in (
            _special_token_id(tokenizer, "bos_id", "bos"),
            _special_token_id(tokenizer, "eos_id", "eos", "eod"),
            _special_token_id(tokenizer, "pad_id", "pad"),
        )
        if token_id is not None
    }

    if not special_token_ids:
        return _detokenize_with_special_tokens(tokenizer, token_ids)

    full_text = _detokenize_with_special_tokens(tokenizer, token_ids)
    preserved_special_tokens = []
    for token_id in token_ids:
        if token_id not in special_token_ids:
            continue
        try:
            special_text = _detokenize_with_special_tokens(tokenizer, [token_id])
        except Exception:
            special_text = None
        preserved_special_tokens.append(bool(special_text and special_text in full_text))
    if preserved_special_tokens and all(preserved_special_tokens):
        return full_text

    pieces = []
    normal_span = []

    def flush_normal_span() -> None:
        if normal_span:
            pieces.append(_detokenize_with_special_tokens(tokenizer, normal_span))
            normal_span.clear()

    for token_id in token_ids:
        if token_id not in special_token_ids:
            normal_span.append(token_id)
            continue

        flush_normal_span()
        try:
            special_text = _detokenize_with_special_tokens(tokenizer, [token_id])
        except Exception:
            special_text = None
        pieces.append(special_text or _special_token_piece(tokenizer, token_id) or "")

    flush_normal_span()
    return "".join(pieces)
