#!/usr/bin/env python3
"""Prepare a small FineWeb-Edu JSONL slice for lm_eval perplexity runs."""

import argparse
import json
import os
from pathlib import Path


DEFAULT_SOURCE = (
    "/iopsstor/scratch/cscs/tr1eder/cler/_research/results/data/fineweb_edu/"
    "fineweb_edu_15b_llama2_tc/fineweb_edu_15b_llama2_tc.jsonl"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-jsonl",
        default=os.environ.get("FINEWEB_EDU_SOURCE_JSONL", DEFAULT_SOURCE),
        help="FineWeb-Edu JSONL source file.",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("FINEWEB_EDU_EVAL_OUTPUT"),
        help=(
            "Output JSONL path. Defaults to "
            "_research/results/eval/fineweb_edu/fineweb_edu_15b_tail_<docs>.jsonl."
        ),
    )
    parser.add_argument(
        "--docs",
        type=int,
        default=int(os.environ.get("FINEWEB_EDU_EVAL_DOCS", "10000")),
        help="Number of documents to keep from the validation tail.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=float(os.environ.get("FINEWEB_EDU_VALIDATION_FRACTION", "0.01")),
        help="Fraction of the source documents treated as the validation tail.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild an existing slice.")
    return parser.parse_args()


def manifest_path(path):
    return Path(str(path) + ".manifest.json")


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as fin:
        return json.load(fin)


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)
        fout.write("\n")


def output_is_current(output, source, docs, validation_fraction, source_manifest):
    manifest = manifest_path(output)
    if not output.exists() or not manifest.exists():
        return False

    data = read_json(manifest)
    return (
        data.get("status") == "complete"
        and data.get("source_jsonl") == str(source)
        and data.get("source_manifest") == str(manifest_path(source))
        and data.get("source_written_docs") == source_manifest.get("written_docs")
        and data.get("selected_docs") == docs
        and data.get("validation_fraction") == validation_fraction
    )


def tail_lines(path, count, chunk_size=8 * 1024 * 1024):
    """Return the last count JSONL lines using bounded reverse reads."""
    if count <= 0:
        return []

    blocks = []
    newline_count = 0
    with path.open("rb") as fin:
        fin.seek(0, os.SEEK_END)
        position = fin.tell()
        while position > 0 and newline_count <= count:
            read_size = min(chunk_size, position)
            position -= read_size
            fin.seek(position)
            block = fin.read(read_size)
            blocks.append(block)
            newline_count += block.count(b"\n")

    data = b"".join(reversed(blocks))
    return data.splitlines()[-count:]


def main():
    args = parse_args()
    source = Path(args.source_jsonl)
    if not source.is_file():
        raise SystemExit("source JSONL not found: {}".format(source))

    source_manifest_path = manifest_path(source)
    if not source_manifest_path.is_file():
        raise SystemExit("source manifest not found: {}".format(source_manifest_path))

    if args.docs <= 0:
        raise SystemExit("--docs must be positive")
    if not (0.0 < args.validation_fraction <= 1.0):
        raise SystemExit("--validation-fraction must be in (0, 1]")

    output = Path(
        args.output
        or "_research/results/eval/fineweb_edu/fineweb_edu_15b_tail_{}.jsonl".format(
            args.docs
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    source_manifest = read_json(source_manifest_path)
    source_docs = int(source_manifest["written_docs"])
    validation_docs = max(1, int(source_docs * args.validation_fraction))
    selected_docs = min(args.docs, validation_docs)

    if output.exists() and not args.overwrite:
        if output_is_current(output, source, selected_docs, args.validation_fraction, source_manifest):
            print("Reusing FineWeb-Edu eval slice: {}".format(output))
            return 0
        raise SystemExit(
            "{} exists but does not match the requested slice. Pass --overwrite.".format(output)
        )

    lines = tail_lines(source, selected_docs)
    if len(lines) != selected_docs:
        raise SystemExit(
            "expected {} tail docs from {}, got {}".format(selected_docs, source, len(lines))
        )

    with output.open("wb") as fout:
        for line in lines:
            if not line.strip():
                continue
            fout.write(line)
            fout.write(b"\n")

    write_json(
        manifest_path(output),
        {
            "status": "complete",
            "source_jsonl": str(source),
            "source_manifest": str(source_manifest_path),
            "source_dataset": source_manifest.get("dataset"),
            "source_split": source_manifest.get("split"),
            "source_written_docs": source_docs,
            "source_written_tokens": source_manifest.get("written_tokens"),
            "validation_fraction": args.validation_fraction,
            "validation_docs": validation_docs,
            "selection": "last_docs_within_validation_tail",
            "selected_docs": selected_docs,
            "output": str(output),
        },
    )

    print("Wrote FineWeb-Edu eval slice: {}".format(output))
    print("Selected docs: {} from validation tail of {}".format(selected_docs, validation_docs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
