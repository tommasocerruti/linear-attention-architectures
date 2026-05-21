"""Local lm_eval runtime shims for CLER eval jobs.

The full local-completions run can issue tens of thousands of remote tokenizer
HTTP calls before generation starts. Werkzeug closes these localhost
connections quickly, so an unconstrained client can exhaust ephemeral source
ports and fail with ``Cannot assign requested address``. Keep using the remote
tokenizer endpoints, but pace uncached calls and cache exact tokenization
results inside the eval process.
"""

from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict
from functools import wraps


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _install_remote_tokenizer_patch() -> None:
    try:
        import requests
        from lm_eval.utils import RemoteTokenizer
    except Exception:
        return

    if getattr(RemoteTokenizer, "_cler_remote_tokenizer_patch", False):
        return

    delay_seconds = _float_env("LM_EVAL_REMOTE_TOKENIZER_DELAY_SECONDS", 0.005)
    cache_size = _int_env("LM_EVAL_REMOTE_TOKENIZER_CACHE_SIZE", 50000)
    original_request = RemoteTokenizer._request_with_retries
    original_encode = RemoteTokenizer.encode
    original_decode = RemoteTokenizer.decode

    def _cache_get(cache: OrderedDict, key):
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value

    def _cache_put(cache: OrderedDict, key, value):
        if cache_size <= 0:
            return
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > cache_size:
            cache.popitem(last=False)

    def _request_with_retries(self, method, url, **kwargs):
        last_exc = None
        timeout = kwargs.pop("timeout", self.timeout)
        for attempt in range(self.max_retries):
            try:
                resp = self.session.request(
                    method,
                    url,
                    timeout=timeout,
                    verify=self.cert_config,
                    **kwargs,
                )
                resp.raise_for_status()
                if delay_seconds:
                    time.sleep(delay_seconds)
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(max(delay_seconds, min(4.0, 0.5 * (attempt + 1))))
        raise RuntimeError(
            f"RemoteTokenizer: {method} {url} failed after {self.max_retries} attempts: {last_exc}"
        )

    @wraps(original_encode)
    def encode(self, text: str) -> list[int]:
        if cache_size <= 0:
            return original_encode(self, text)
        cache = getattr(self, "_cler_encode_cache", None)
        if cache is None:
            cache = OrderedDict()
            self._cler_encode_cache = cache
        key = text
        cached = _cache_get(cache, key)
        if cached is not None:
            return list(cached)
        tokens = list(original_encode(self, text))
        _cache_put(cache, key, tuple(tokens))
        return tokens

    @wraps(original_decode)
    def decode(self, tokens: list[int]) -> str:
        if cache_size <= 0:
            return original_decode(self, tokens)
        cache = getattr(self, "_cler_decode_cache", None)
        if cache is None:
            cache = OrderedDict()
            self._cler_decode_cache = cache
        key = tuple(tokens)
        cached = _cache_get(cache, key)
        if cached is not None:
            return cached
        text = original_decode(self, tokens)
        _cache_put(cache, key, text)
        return text

    RemoteTokenizer._request_with_retries = _request_with_retries
    RemoteTokenizer.encode = encode
    RemoteTokenizer.decode = decode
    RemoteTokenizer._cler_remote_tokenizer_patch = True
    sys.stderr.write(
        "[cler] patched lm_eval RemoteTokenizer "
        f"(delay={delay_seconds}s, cache_size={cache_size})\n"
    )


_install_remote_tokenizer_patch()
