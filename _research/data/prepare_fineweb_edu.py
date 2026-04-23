#!/usr/bin/env python3
"""Create a deterministic FineWeb-Edu JSONL prefix for Megatron preprocessing.

The script streams documents in dataset order and stops at the first document
boundary whose cumulative token count reaches the requested budget. For the
FineWeb-Edu 100BT shuffled dataset this gives a deterministic 15B-token prefix
with no local shuffle step.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="HuggingFaceFW/fineweb_edu_100BT-shuffled",
        help="Hugging Face dataset id to stream.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional Hugging Face dataset config/name.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to stream.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path. Existing completed outputs are reused.",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=15_000_000_000,
        help="Token budget. Stops after the document that reaches this count.",
    )
    parser.add_argument("--text-key", default="text", help="Dataset field containing text.")
    parser.add_argument(
        "--token-count-key",
        default="token_count",
        help=(
            "Dataset token-count field. Set to an empty string to force local "
            "tokenization for the budget."
        ),
    )
    parser.add_argument(
        "--fallback-tokenizer",
        default="gpt2",
        help=(
            "Transformers tokenizer name used when token_count is missing and "
            "--sentencepiece-tokenizer-model is not set."
        ),
    )
    parser.add_argument(
        "--sentencepiece-tokenizer-model",
        default=None,
        help="SentencePiece model path used for fallback token counting.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional cap for debugging/smoke-test subsets.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10_000,
        help="Progress logging interval in written documents.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing incomplete or complete output.",
    )
    return parser.parse_args()


def load_dataset_stream(args: argparse.Namespace) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Run _research/launch/install_python_deps.sh first."
        ) from exc

    kwargs = {
        "path": args.dataset,
        "split": args.split,
        "streaming": True,
    }
    if args.config is not None:
        kwargs["name"] = args.config
    return load_dataset(**kwargs)


def load_sentencepiece_tokenizer(model_path: str) -> Any:
    try:
        import sentencepiece
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sentencepiece. Run _research/launch/install_python_deps.sh first."
        ) from exc

    tokenizer = sentencepiece.SentencePieceProcessor()
    if not tokenizer.Load(model_path):
        raise SystemExit(f"Failed to load SentencePiece tokenizer model: {model_path}")
    return tokenizer


def load_fallback_tokenizer(name: str, sentencepiece_model: Optional[str] = None) -> Any:
    if sentencepiece_model:
        return load_sentencepiece_tokenizer(sentencepiece_model)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Run _research/launch/install_python_deps.sh first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(name)
    return tokenizer


def manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def existing_output_is_complete(output: Path, args: argparse.Namespace) -> bool:
    manifest = manifest_path(output)
    if not output.exists() or not manifest.exists():
        return False

    with manifest.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    return (
        data.get("status") == "complete"
        and data.get("dataset") == args.dataset
        and data.get("config") == args.config
        and data.get("split") == args.split
        and data.get("target_tokens") == args.target_tokens
    )


def get_token_count(
    sample: Dict[str, Any],
    token_count_key: str,
    fallback_tokenizer: str,
    sentencepiece_model: Optional[str],
    text: str,
    tokenizer: Optional[Any],
) -> Tuple[int, Optional[Any], str]:
    if token_count_key and token_count_key in sample and sample[token_count_key] is not None:
        return int(sample[token_count_key]), tokenizer, f"dataset:{token_count_key}"

    if tokenizer is None:
        tokenizer = load_fallback_tokenizer(fallback_tokenizer, sentencepiece_model)
    if hasattr(tokenizer, "EncodeAsIds"):
        return (
            len(tokenizer.EncodeAsIds(text)),
            tokenizer,
            f"fallback:sentencepiece:{sentencepiece_model}",
        )
    return len(tokenizer.encode(text)), tokenizer, f"fallback:{fallback_tokenizer}"


def write_manifest(
    output: Path,
    args: argparse.Namespace,
    status: str,
    written_docs: int,
    written_tokens: int,
    token_count_source: str,
    elapsed_seconds: float,
) -> None:
    manifest = {
        "status": status,
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "target_tokens": args.target_tokens,
        "written_docs": written_docs,
        "written_tokens": written_tokens,
        "text_key": args.text_key,
        "token_count_source": token_count_source,
        "output": str(output),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    with manifest_path(output).open("w", encoding="utf-8") as fout:
        json.dump(manifest, fout, indent=2, sort_keys=True)
        fout.write("\n")


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.overwrite:
        if existing_output_is_complete(output, args):
            print(f"Reusing completed output: {output}", file=sys.stderr)
            return 0
        raise SystemExit(
            f"{output} already exists but is not a matching completed output. "
            "Pass --overwrite to rebuild it."
        )

    if args.overwrite:
        output.unlink(missing_ok=True)
        manifest_path(output).unlink(missing_ok=True)

    dataset = load_dataset_stream(args)
    tokenizer = None
    token_count_source = ""
    written_docs = 0
    written_tokens = 0
    start = time.time()

    with output.open("w", encoding="utf-8") as fout:
        for sample in dataset:
            text = sample.get(args.text_key)
            if not text:
                continue

            token_count, tokenizer, source = get_token_count(
                sample,
                args.token_count_key,
                args.fallback_tokenizer,
                args.sentencepiece_tokenizer_model,
                text,
                tokenizer,
            )
            token_count_source = token_count_source or source

            fout.write(json.dumps({"text": text}, ensure_ascii=False))
            fout.write("\n")
            written_docs += 1
            written_tokens += token_count

            if written_docs % args.log_interval == 0:
                elapsed = max(time.time() - start, 1e-6)
                docs_per_second = written_docs / elapsed
                pct = min(100.0, 100.0 * written_tokens / args.target_tokens)
                print(
                    f"[{pct:6.2f}%] docs={written_docs:,} "
                    f"tokens={written_tokens:,} docs/s={docs_per_second:,.1f}",
                    file=sys.stderr,
                    flush=True,
                )

            if written_tokens >= args.target_tokens:
                break
            if args.max_documents is not None and written_docs >= args.max_documents:
                break

    elapsed = time.time() - start
    status = "complete" if written_tokens >= args.target_tokens else "partial"
    write_manifest(
        output,
        args,
        status=status,
        written_docs=written_docs,
        written_tokens=written_tokens,
        token_count_source=token_count_source,
        elapsed_seconds=elapsed,
    )
    print(
        f"Wrote {written_docs:,} docs / {written_tokens:,} tokens to {output} ({status}).",
        file=sys.stderr,
    )
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
