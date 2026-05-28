# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""OpenAI-compatible score-only server for lm-eval loglikelihood requests.

This entrypoint intentionally does not use Megatron's autoregressive inference
engine.  It runs ordinary full-sequence forward passes and formats next-token
log probabilities in the shape expected by ``lm_eval``'s ``local-completions``
model.  That keeps likelihood/perplexity eval available for model variants
whose layers do not implement incremental inference caches.
"""

import os
import sys
import threading
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

try:
    from contextlib import nullcontext
except ImportError:
    class nullcontext:  # type: ignore
        def __init__(self, enter_result=None):
            self.enter_result = enter_result

        def __enter__(self):
            return self.enter_result

        def __exit__(self, *excinfo):
            return None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)


def build_lm_eval_completion_choice(
    *,
    index: int,
    text: str,
    token_texts: List[str],
    text_offsets: List[int],
    next_token_logprobs: List[float],
    next_token_top_logprobs: Optional[List[Dict[str, float]]],
) -> Dict[str, Any]:
    """Format scores to match lm-eval's local-completions slicing contract.

    For an echoed prompt of N tokens, next-token logprobs exist for tokens
    1..N-1.  ``lm_eval`` requests ``max_tokens=1`` and drops the final generated
    token via ``token_logprobs[ctxlen:-1]``.  This score-only server appends a
    dummy terminal item so the same slice selects exactly the continuation.
    """

    if len(text_offsets) != len(token_texts):
        raise ValueError("text_offsets and token_texts must have the same length")

    expected_score_count = max(len(token_texts) - 1, 0)
    if len(next_token_logprobs) != expected_score_count:
        raise ValueError(
            "next_token_logprobs must have length len(token_texts) - 1, "
            f"got {len(next_token_logprobs)} for {len(token_texts)} tokens"
        )
    if (
        next_token_top_logprobs is not None
        and len(next_token_top_logprobs) != expected_score_count
    ):
        raise ValueError(
            "next_token_top_logprobs must have length len(token_texts) - 1, "
            f"got {len(next_token_top_logprobs)} for {len(token_texts)} tokens"
        )

    token_logprobs = [None]
    token_logprobs.extend(float(value) for value in next_token_logprobs)
    token_logprobs.append(0.0)

    tokens = list(token_texts)
    tokens.append("")

    offsets = [int(value) for value in text_offsets]
    offsets.append(len(text))

    top_logprobs = None
    if next_token_top_logprobs is not None:
        top_logprobs = [None]
        top_logprobs.extend(next_token_top_logprobs)
        top_logprobs.append({"": 0.0})

    return {
        "index": index,
        "text": text,
        "logprobs": {
            "token_logprobs": token_logprobs,
            "tokens": tokens,
            "text_offset": offsets,
            "top_logprobs": top_logprobs,
        },
    }


def _coerce_prompt_items(prompt: Any, tokenizer: Any) -> List[Tuple[str, List[int]]]:
    from megatron.core.inference.text_generation_server.tokenizer_api import (
        detokenize_for_api,
        tokenize_for_api,
    )

    if isinstance(prompt, str):
        return [(prompt, tokenize_for_api(tokenizer, prompt, add_special_tokens=False))]

    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a string, token list, or non-empty list of prompts")

    if all(isinstance(item, str) for item in prompt):
        return [
            (item, tokenize_for_api(tokenizer, item, add_special_tokens=False))
            for item in prompt
        ]

    if all(isinstance(item, int) for item in prompt):
        token_ids = list(prompt)
        return [(detokenize_for_api(tokenizer, token_ids), token_ids)]

    if all(isinstance(item, list) and all(isinstance(token, int) for token in item) for item in prompt):
        return [(detokenize_for_api(tokenizer, item), list(item)) for item in prompt]

    raise ValueError("prompt list must contain only strings, ints, or lists of ints")


def _token_text(tokenizer: Any, token_id: int) -> str:
    from megatron.core.inference.text_generation_server.tokenizer_api import detokenize_for_api

    return detokenize_for_api(tokenizer, [int(token_id)])


def _token_offsets(
    tokenizer: Any, token_ids: List[int], text: str, token_texts: List[str]
) -> List[int]:
    if hasattr(tokenizer, "offsets"):
        offsets = tokenizer.offsets(token_ids, text)
        offsets = [int(value) for value in offsets]
        if len(offsets) == len(token_ids):
            return offsets

    offsets = []
    cursor = 0
    for token_text in token_texts:
        offsets.append(cursor)
        cursor += len(token_text)
    return offsets


def _json_error(message: str, status: int = 400):
    from flask import jsonify

    return jsonify({"error": message}), status


class LoglikelihoodScorer:
    """Batch scorer that maps prompts to OpenAI completions responses."""

    def __init__(self, model: Any, tokenizer: Any, max_seq_length: int):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.lock = threading.Lock()

    def score(self, prompt_items: List[Tuple[str, List[int]]], top_n: int) -> List[Dict[str, Any]]:
        import torch
        import torch.nn.functional as F

        if not prompt_items:
            raise ValueError("at least one prompt is required")
        if top_n < 0:
            raise ValueError("logprobs must be non-negative")
        if top_n > 10:
            raise ValueError("logprobs > 10 is not supported")

        lengths = [len(token_ids) for _, token_ids in prompt_items]
        max_len = max(lengths, default=0)
        if max_len > self.max_seq_length:
            raise ValueError(
                f"prompt has {max_len} tokens, exceeding max sequence length {self.max_seq_length}"
            )

        if max_len == 0:
            return [
                build_lm_eval_completion_choice(
                    index=index,
                    text=text,
                    token_texts=[],
                    text_offsets=[],
                    next_token_logprobs=[],
                    next_token_top_logprobs=[] if top_n else None,
                )
                for index, (text, _) in enumerate(prompt_items)
            ]

        device = torch.device("cuda", torch.cuda.current_device())
        pad_id = getattr(self.tokenizer, "eod", None)
        if not isinstance(pad_id, int):
            pad_id = getattr(self.tokenizer, "eos_id", None)
        if not isinstance(pad_id, int):
            pad_id = 0

        tokens = torch.full((len(prompt_items), max_len), pad_id, dtype=torch.long, device=device)
        for row, (_, token_ids) in enumerate(prompt_items):
            if token_ids:
                tokens[row, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long, device=device)

        position_ids = torch.arange(max_len, dtype=torch.long, device=device).unsqueeze(0)
        position_ids = position_ids.expand(len(prompt_items), -1).contiguous()

        attention_mask = torch.triu(
            torch.ones((1, 1, max_len, max_len), dtype=torch.bool, device=device),
            diagonal=1,
        )
        padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= torch.tensor(
            lengths, dtype=torch.long, device=device
        ).unsqueeze(1)

        with self.lock, torch.inference_mode():
            logits = self.model(
                tokens,
                position_ids,
                attention_mask,
                labels=None,
                runtime_gather_output=True,
                padding_mask=padding_mask,
            )
            log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)

        choices = []
        for index, (text, token_ids) in enumerate(prompt_items):
            token_texts = [_token_text(self.tokenizer, token_id) for token_id in token_ids]
            offsets = _token_offsets(self.tokenizer, token_ids, text, token_texts)

            seq_len = len(token_ids)
            if seq_len <= 1:
                choices.append(
                    build_lm_eval_completion_choice(
                        index=index,
                        text=text,
                        token_texts=token_texts,
                        text_offsets=offsets,
                        next_token_logprobs=[],
                        next_token_top_logprobs=[] if top_n else None,
                    )
                )
                continue

            row_log_probs = log_probs[index, : seq_len - 1]
            target_ids = tokens[index, 1:seq_len]
            next_token_logprobs = row_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

            next_token_top_logprobs = None
            if top_n:
                top_values, top_indices = torch.topk(row_log_probs, k=top_n, dim=-1)
                next_token_top_logprobs = []
                for values, indices in zip(top_values.tolist(), top_indices.tolist()):
                    next_token_top_logprobs.append(
                        {
                            _token_text(self.tokenizer, token_id): float(value)
                            for token_id, value in zip(indices, values)
                        }
                    )

            choices.append(
                build_lm_eval_completion_choice(
                    index=index,
                    text=text,
                    token_texts=token_texts,
                    text_offsets=offsets,
                    next_token_logprobs=[float(value) for value in next_token_logprobs.tolist()],
                    next_token_top_logprobs=next_token_top_logprobs,
                )
            )

        return choices


class ScoringHttpServer:
    def __init__(self, scorer: LoglikelihoodScorer):
        from flask import Flask, jsonify, request
        from megatron.core.inference.text_generation_server.tokenizer_api import (
            detokenize_for_api,
            get_tokenizer_info,
            tokenize_for_api,
        )

        self.app = Flask(__name__, static_url_path="")
        tokenizer = scorer.tokenizer

        @self.app.route("/completions", methods=["POST"])
        @self.app.route("/v1/completions", methods=["POST"])
        def completions():
            req = request.get_json(force=True)
            if not req or "prompt" not in req:
                return _json_error("Missing 'prompt' field")

            max_tokens = int(req.get("max_tokens", 0))
            echo = bool(req.get("echo", False))
            top_n = int(req.get("logprobs", 0))
            best_of = int(req.get("best_of", 1))
            n = int(req.get("n", 1))

            if not echo:
                return _json_error("score-only completions require echo=true")
            if top_n <= 0:
                return _json_error("score-only completions require logprobs >= 1")
            if max_tokens not in (0, 1):
                return _json_error("score-only completions support only max_tokens 0 or 1")
            if best_of > 1:
                return _json_error("best_of > 1 is not supported")
            if n > 1:
                return _json_error("n > 1 is not supported")

            try:
                prompt_items = _coerce_prompt_items(req["prompt"], tokenizer)
                choices = scorer.score(prompt_items, top_n=top_n)
            except (TypeError, ValueError) as exc:
                return _json_error(str(exc))
            except Exception as exc:
                return _json_error(f"error scoring prompt: {exc}", status=500)

            return jsonify({"choices": choices})

        @self.app.route("/tokenizer_info", methods=["GET"])
        @self.app.route("/v1/tokenizer_info", methods=["GET"])
        def tokenizer_info():
            return jsonify(get_tokenizer_info(tokenizer))

        @self.app.route("/tokenize", methods=["POST"])
        @self.app.route("/v1/tokenize", methods=["POST"])
        def tokenize():
            req = request.get_json(force=True)
            prompt = req.get("prompt") if req else None
            if prompt is None:
                return _json_error("Missing 'prompt' field")

            try:
                token_ids = tokenize_for_api(
                    tokenizer, prompt, add_special_tokens=bool(req.get("add_special_tokens", False))
                )
            except (TypeError, ValueError) as exc:
                return _json_error(str(exc))
            except Exception as exc:
                return _json_error(f"Error tokenizing prompt: {exc}", status=500)

            return jsonify({"tokens": token_ids})

        @self.app.route("/detokenize", methods=["POST"])
        @self.app.route("/v1/detokenize", methods=["POST"])
        def detokenize():
            req = request.get_json(force=True)
            token_ids = req.get("tokens") if req else None
            if token_ids is None:
                return _json_error("Missing 'tokens' field")

            try:
                prompt = detokenize_for_api(tokenizer, token_ids)
            except (TypeError, ValueError) as exc:
                return _json_error(str(exc))
            except Exception as exc:
                return _json_error(f"Error detokenizing tokens: {exc}", status=500)

            return jsonify({"prompt": prompt})

    def run(self, host: str, port: int) -> None:
        self.app.run(host, threaded=True, debug=False, port=port)


def add_scoring_server_args(parser):
    from megatron.post_training.arguments import add_modelopt_args

    group = parser.add_argument_group(title="lm-eval scoring server")
    group.add_argument("--port", type=int, default=5000, help="port to bind")
    group.add_argument("--host", type=str, default="0.0.0.0", help="host interface to bind")
    group.add_argument(
        "--scoring-max-seq-length",
        type=int,
        default=None,
        help="maximum tokenized prompt length accepted by the scorer",
    )
    add_modelopt_args(parser)
    return parser


def main(model_type: str = "gpt") -> None:
    import torch

    from gpt_builders import gpt_builder
    from mamba_builders import mamba_builder
    from megatron.training import get_args, get_model, get_tokenizer, print_rank_0
    from megatron.training.arguments import parse_and_validate_args
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.initialize import initialize_megatron
    from model_provider import model_provider

    parse_and_validate_args(
        extra_args_provider=add_scoring_server_args,
        args_defaults={
            "no_load_rng": True,
            "no_load_optim": True,
            "micro_batch_size": 1,
            "exit_on_missing_checkpoint": True,
        },
    )
    initialize_megatron()
    args = get_args()

    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise RuntimeError("score-only lm_eval server currently supports single-process eval only")

    if args.num_layers_per_virtual_pipeline_stage is not None:
        raise RuntimeError("interleaved pipeline schedule is not supported for scoring")

    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for lm-eval scoring.")
    args.exit_on_missing_checkpoint = True

    load_context = nullcontext()
    if args.fp8:
        from transformer_engine.pytorch.fp8 import fp8_model_init

        load_context = fp8_model_init()

    with load_context:
        if model_type == "gpt":
            model_builder = gpt_builder
        elif model_type == "mamba":
            model_builder = mamba_builder
        else:
            raise ValueError(f"Invalid model provider {model_type}")
        model = get_model(partial(model_provider, model_builder), wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None, strict=False)

    assert len(model) == 1, "score-only server expects one local model chunk"
    model = model[0]
    model.eval()

    max_seq_length = args.scoring_max_seq_length
    if max_seq_length is None:
        max_seq_length = getattr(
            args, "inference_max_seq_length", getattr(args, "inference_max_sequence_length", 4096)
        )

    scorer = LoglikelihoodScorer(model, get_tokenizer(), max_seq_length=max_seq_length)
    server = ScoringHttpServer(scorer)
    server.run(args.host, port=args.port)


if __name__ == "__main__":
    main(model_type="gpt")
