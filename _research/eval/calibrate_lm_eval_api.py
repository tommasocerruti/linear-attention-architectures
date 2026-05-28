#!/usr/bin/env python3
"""Probe the local-completions loglikelihood contract used by lm_eval."""

import argparse
import json
from collections import namedtuple
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ContinuationProbe = namedtuple("ContinuationProbe", ["name", "context", "continuations"])


PROBES = (
    ContinuationProbe(
        "france-capital",
        "The capital of France is",
        (" Paris", " banana"),
    ),
    ContinuationProbe(
        "italy-capital",
        "The capital of Italy is",
        (" Rome", " spoon"),
    ),
    ContinuationProbe(
        "simple-grammar",
        "The quick brown fox",
        (" jumps over the lazy dog.", " purple the quickly."),
    ),
    ContinuationProbe(
        "common-phrase",
        "Once upon a time",
        (" there was a", " electrical sandwich"),
    ),
)


def post_json(base_url, path, payload):
    request = Request(
        "{}{}".format(base_url, path),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP {} from {}: {}".format(error.code, path, body))


def encode(base_url, text):
    payload = post_json(base_url, "/tokenize", {"prompt": text, "add_special_tokens": False})
    return list(payload["tokens"])


def decode(base_url, tokens):
    payload = post_json(base_url, "/detokenize", {"tokens": tokens})
    return str(payload["prompt"])


def encode_pair(base_url, context, continuation):
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]

    whole = encode(base_url, context + continuation)
    context_tokens = encode(base_url, context)
    return context_tokens, whole[len(context_tokens) :]


def score_continuation(base_url, context, continuation):
    context_tokens, continuation_tokens = encode_pair(base_url, context, continuation)
    request_tokens = context_tokens + continuation_tokens
    request_text = decode(base_url, request_tokens)
    ctxlen = len(context_tokens)

    response = post_json(
        base_url,
        "/v1/completions",
        {
            "model": "cler",
            "prompt": request_text,
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": 1,
            "seed": 1234,
            "echo": True,
        },
    )
    choice = sorted(response["choices"], key=lambda item: item["index"])[0]
    logprobs = choice["logprobs"]
    token_logprobs = logprobs["token_logprobs"]
    top_logprobs = logprobs["top_logprobs"]

    scored_logprobs = token_logprobs[ctxlen:-1]
    scored_top_logprobs = top_logprobs[ctxlen:-1]
    scored_tokens = logprobs["tokens"][ctxlen:-1]
    scored_offsets = logprobs["text_offset"][ctxlen:-1]
    is_greedy = all(
        token_logprob == max(top_logprob.values())
        for token_logprob, top_logprob in zip(scored_logprobs, scored_top_logprobs)
    )

    return {
        "context": context,
        "continuation": continuation,
        "ctxlen": ctxlen,
        "request_token_count": len(request_tokens),
        "response_token_count": len(logprobs["tokens"]),
        "token_logprob_count": len(token_logprobs),
        "top_logprob_count": len(top_logprobs),
        "text_offset_count": len(logprobs["text_offset"]),
        "request_text": request_text,
        "score": sum(scored_logprobs),
        "is_greedy": is_greedy,
        "scored_tokens": scored_tokens,
        "scored_offsets": scored_offsets,
        "scored_logprobs": scored_logprobs,
        "scored_top_logprobs": scored_top_logprobs,
        "boundary_tokens": logprobs["tokens"][max(0, ctxlen - 3) : ctxlen + 4],
        "boundary_logprobs": token_logprobs[max(0, ctxlen - 3) : ctxlen + 4],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    args = parser.parse_args()

    all_results = []
    for probe in PROBES:
        results = [
            score_continuation(args.base_url, probe.context, continuation)
            for continuation in probe.continuations
        ]
        winner = max(range(len(results)), key=lambda idx: results[idx]["score"])
        all_results.append({"name": probe.name, "winner": winner, "results": results})

    print(json.dumps(all_results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
