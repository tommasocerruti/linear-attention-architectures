#!/usr/bin/env python3
"""Parse quick-run Megatron logs and generate comparison plots.

Assumptions:
- Logs are text files from Megatron quick smoke runs on Clariden.
- Per-iteration metrics appear on lines like:
  ``iteration      200/     200 | ... | lm loss: 5.28E+00 | ...``
- Tokens/sec/GPU appears on lines like:
  ``[apertus] iter   200 | tokens/s/GPU: 92078.3``
- Memory snapshots appear on lines like:
  ``[Rank 0] (after 2 iterations) memory (MB) | allocated: ... | max allocated: ...``

The script keeps a small manifest of the validated runs that should appear in
the final plots, but it also auto-scans the run directory so failed or
superseded attempts can be summarized in the markdown output.

Recommended invocation on this machine:
  /users/course_00252/miniconda3/bin/python _research/plotting/plot_quick_results.py
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt


RUNS_DIR = Path("/iopsstor/scratch/cscs/course_00252/cler/_research/results/runs")
REPO_ROOT = Path("/users/course_00252/cler")
PLOTS_DIR = REPO_ROOT / "tracker" / "week1" / "plots"
SUMMARY_MD = REPO_ROOT / "tracker" / "week1" / "23_04_2026_results.md"
TRACKER_MD = REPO_ROOT / "tracker" / "week1" / "23_04_2026.md"

PRIMARY_MANIFEST = [
    {
        "label": "Softmax AdamW",
        "path": RUNS_DIR / "quick-softmax-1gpu-climbmix-1928332.log",
        "architecture": "Softmax",
        "optimizer": "AdamW",
    },
    {
        "label": "GDN AdamW",
        "path": RUNS_DIR / "quick-gdn-1gpu-climbmix-1928665.log",
        "architecture": "GDN",
        "optimizer": "AdamW",
    },
    {
        "label": "Softmax Muon",
        "path": RUNS_DIR / "quick-softmax-1gpu-muon-climbmix-1929341.log",
        "architecture": "Softmax",
        "optimizer": "Muon",
    },
    {
        "label": "GDN Muon",
        "path": RUNS_DIR / "quick-gdn-1gpu-muon-climbmix-1929351.log",
        "architecture": "GDN",
        "optimizer": "Muon",
    },
]

SUPERSEDED_SUCCESSFUL_LOGS = {
    "quick-softmax-1gpu-muon-climbmix-1928987.log": "Earlier successful softmax+Muon duplicate; latest validated rerun is 1929341.",
    "quick-gdn-1gpu-muon-climbmix-1929094.log": "Earlier successful GDN+Muon duplicate; latest validated rerun is 1929351.",
}

PREFERRED_FAILED_LOGS = {
    "quick-gdn-1gpu-climbmix-1928333.log": "Failed before iter 1 due to missing FLA layout before the dependency fix.",
    "quick-softmax-1gpu-muon-climbmix-1929331.log": "Failed before training because the 1-GPU Muon smoke still used the distributed-optimizer path.",
    "quick-gdn-1gpu-muon-climbmix-1929347.log": "Failed at step 1 in Muon's QKV split reshape path before adding --muon-no-split-qkv.",
}

ITER_RE = re.compile(
    r"iteration\s+(?P<iteration>\d+)/\s*(?P<total>\d+)\s+\|(?P<body>.+?)\|$"
)
FIELD_RE = {
    "loss": re.compile(r"lm loss:\s+([0-9.eE+-]+)"),
    "lr": re.compile(r"learning rate:\s+([0-9.eE+-]+)"),
    "elapsed_ms": re.compile(r"elapsed time per iteration \(ms\):\s+([0-9.eE+-]+)"),
    "tflops": re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s+([0-9.eE+-]+)"),
    "skipped": re.compile(r"number of skipped iterations:\s+(\d+)"),
    "nan_iters": re.compile(r"number of nan iterations:\s+(\d+)"),
    "grad_norm": re.compile(r"grad norm:\s+([0-9.eE+-]+)"),
    "params_norm": re.compile(r"params norm:\s+([0-9.eE+-]+)"),
}
TOKENS_RE = re.compile(r"\[apertus\]\s+iter\s+(?P<iteration>\d+)\s+\|\s+tokens/s/GPU:\s+(?P<tokens>[0-9.eE+-]+)")
MEMORY_RE = re.compile(
    r"\(after\s+(?P<iteration>\d+)\s+iterations\)\s+memory \(MB\) \| "
    r"allocated:\s+(?P<allocated>[0-9.eE+-]+) \| "
    r"max allocated:\s+(?P<max_allocated>[0-9.eE+-]+) \| "
    r"reserved:\s+(?P<reserved>[0-9.eE+-]+) \| "
    r"max reserved:\s+(?P<max_reserved>[0-9.eE+-]+)"
)
PARAMS_RE = re.compile(r"number of parameters on .*: (?P<params>\d+)")
OPT_RE = re.compile(r"optimizer\s+\.+\s+(?P<optimizer>\S+)")
ATTN_RE = re.compile(r"experimental_attention_variant\s+\.+\s+(?P<variant>\S+)")


@dataclass
class IterationPoint:
    iteration: int
    total_iterations: Optional[int] = None
    loss: Optional[float] = None
    lr: Optional[float] = None
    elapsed_ms: Optional[float] = None
    tflops_per_gpu: Optional[float] = None
    tokens_per_sec_gpu: Optional[float] = None
    grad_norm: Optional[float] = None
    params_norm: Optional[float] = None
    skipped_iterations: Optional[int] = None
    nan_iterations: Optional[int] = None
    allocated_mb: Optional[float] = None
    max_allocated_mb: Optional[float] = None
    reserved_mb: Optional[float] = None
    max_reserved_mb: Optional[float] = None


@dataclass
class ParsedRun:
    label: str
    path: Path
    architecture: str
    optimizer_family: str
    iterations: List[IterationPoint] = field(default_factory=list)
    params: Optional[int] = None
    optimizer_name: Optional[str] = None
    attention_variant: Optional[str] = None
    end_time_present: bool = False
    traceback_present: bool = False
    runtime_error_present: bool = False

    @property
    def job_id(self) -> str:
        match = re.search(r"-(\d+)\.log$", self.path.name)
        return match.group(1) if match else "unknown"

    @property
    def completed(self) -> bool:
        return self.end_time_present and bool(self.iterations)

    @property
    def final_iteration(self) -> Optional[IterationPoint]:
        return self.iterations[-1] if self.iterations else None

    @property
    def first_successful_step(self) -> Optional[int]:
        return self.iterations[0].iteration if self.iterations else None

    @property
    def total_iterations(self) -> Optional[int]:
        if not self.iterations:
            return None
        return self.iterations[-1].total_iterations

    @property
    def final_loss(self) -> Optional[float]:
        return self.final_iteration.loss if self.final_iteration else None

    @property
    def final_tokens_per_sec_gpu(self) -> Optional[float]:
        return self.final_iteration.tokens_per_sec_gpu if self.final_iteration else None

    @property
    def mean_tokens_per_sec_gpu_after_warmup(self) -> Optional[float]:
        vals = [
            pt.tokens_per_sec_gpu
            for pt in self.iterations
            if pt.tokens_per_sec_gpu is not None and pt.iteration > 20
        ]
        if not vals:
            vals = [pt.tokens_per_sec_gpu for pt in self.iterations if pt.tokens_per_sec_gpu is not None]
        return mean(vals) if vals else None

    @property
    def max_reserved_mb(self) -> Optional[float]:
        vals = [pt.max_reserved_mb for pt in self.iterations if pt.max_reserved_mb is not None]
        return max(vals) if vals else None

    def status_text(self) -> str:
        if self.completed:
            return "completed"
        if self.traceback_present or self.runtime_error_present:
            return "failed"
        if self.iterations:
            return "partial"
        return "unknown"


def parse_optional_float(body: str, key: str) -> Optional[float]:
    match = FIELD_RE[key].search(body)
    return float(match.group(1)) if match else None


def parse_optional_int(body: str, key: str) -> Optional[int]:
    match = FIELD_RE[key].search(body)
    return int(match.group(1)) if match else None


def parse_run(path: Path, label: str, architecture: str, optimizer_family: str) -> ParsedRun:
    run = ParsedRun(
        label=label,
        path=path,
        architecture=architecture,
        optimizer_family=optimizer_family,
    )
    points: Dict[int, IterationPoint] = {}

    with path.open("r", encoding="utf-8", errors="replace") as fin:
        for raw_line in fin:
            line = raw_line.strip()
            if "END TIME:" in line:
                run.end_time_present = True
            if "Traceback" in line:
                run.traceback_present = True
            if "RuntimeError" in line:
                run.runtime_error_present = True

            if run.params is None:
                match = PARAMS_RE.search(line)
                if match:
                    run.params = int(match.group("params"))

            if run.optimizer_name is None:
                match = OPT_RE.search(line)
                if match:
                    run.optimizer_name = match.group("optimizer")

            if run.attention_variant is None:
                match = ATTN_RE.search(line)
                if match:
                    run.attention_variant = match.group("variant")

            match = ITER_RE.search(line)
            if match:
                iteration = int(match.group("iteration"))
                total = int(match.group("total"))
                body = match.group("body")
                point = points.get(iteration, IterationPoint(iteration=iteration))
                point.total_iterations = total
                point.loss = parse_optional_float(body, "loss")
                point.lr = parse_optional_float(body, "lr")
                point.elapsed_ms = parse_optional_float(body, "elapsed_ms")
                point.tflops_per_gpu = parse_optional_float(body, "tflops")
                point.grad_norm = parse_optional_float(body, "grad_norm")
                point.params_norm = parse_optional_float(body, "params_norm")
                point.skipped_iterations = parse_optional_int(body, "skipped")
                point.nan_iterations = parse_optional_int(body, "nan_iters")
                points[iteration] = point
                continue

            match = TOKENS_RE.search(line)
            if match:
                iteration = int(match.group("iteration"))
                point = points.get(iteration, IterationPoint(iteration=iteration))
                point.tokens_per_sec_gpu = float(match.group("tokens"))
                points[iteration] = point
                continue

            match = MEMORY_RE.search(line)
            if match:
                iteration = int(match.group("iteration"))
                point = points.get(iteration, IterationPoint(iteration=iteration))
                point.allocated_mb = float(match.group("allocated"))
                point.max_allocated_mb = float(match.group("max_allocated"))
                point.reserved_mb = float(match.group("reserved"))
                point.max_reserved_mb = float(match.group("max_reserved"))
                points[iteration] = point

    run.iterations = [points[idx] for idx in sorted(points)]
    return run


def infer_architecture_from_name(name: str) -> str:
    if "softmax" in name:
        return "Softmax"
    if "gdn" in name:
        return "GDN"
    return "Unknown"


def infer_optimizer_from_name(name: str) -> str:
    if "muon" in name:
        return "Muon"
    return "AdamW"


def discover_candidate_logs(run_dir: Path) -> List[Path]:
    return sorted(run_dir.glob("quick-*climbmix-*.log"))


def parse_candidates(candidate_paths: Iterable[Path]) -> List[ParsedRun]:
    runs = []
    for path in candidate_paths:
        runs.append(
            parse_run(
                path=path,
                label=path.stem,
                architecture=infer_architecture_from_name(path.name),
                optimizer_family=infer_optimizer_from_name(path.name),
            )
        )
    return runs


def style_for_run(run: ParsedRun) -> Dict[str, object]:
    color = "#1f77b4" if run.architecture == "Softmax" else "#ff7f0e"
    linestyle = "-" if run.optimizer_family == "AdamW" else "--"
    linewidth = 2.6 if run.optimizer_family == "Muon" else 2.2
    return {"color": color, "linestyle": linestyle, "linewidth": linewidth}


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def annotate_final_points(ax, runs: Iterable[ParsedRun]) -> None:
    for run in runs:
        final = run.final_iteration
        if not final or final.loss is None:
            continue
        ax.annotate(
            f"{run.final_loss:.3f}",
            xy=(final.iteration, final.loss),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            color=style_for_run(run)["color"],
            va="center",
        )


def plot_attention_loss_curves(runs: List[ParsedRun], output_path: Path) -> None:
    attention_runs = [
        run for run in runs if run.optimizer_family == "AdamW" and run.architecture in {"Softmax", "GDN"}
    ]
    fig, ax = plt.subplots()
    for run in attention_runs:
        xs = [pt.iteration for pt in run.iterations if pt.loss is not None]
        ys = [pt.loss for pt in run.iterations if pt.loss is not None]
        ax.plot(xs, ys, label=run.label, **style_for_run(run))

    annotate_final_points(ax, attention_runs)
    ax.set_title("Attention Comparison on Quick ClimbMix Run")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("LM Loss")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_optimizer_loss_curves(runs: List[ParsedRun], output_path: Path) -> None:
    fig, ax = plt.subplots()
    for run in runs:
        xs = [pt.iteration for pt in run.iterations if pt.loss is not None]
        ys = [pt.loss for pt in run.iterations if pt.loss is not None]
        ax.plot(xs, ys, label=run.label, **style_for_run(run))

    annotate_final_points(ax, runs)
    ax.set_title("Optimizer Comparison on Quick ClimbMix Runs")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("LM Loss")
    ax.legend(loc="upper right", frameon=True, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _bar_label(ax, bars, fmt: str) -> None:
    for bar in bars:
        height = bar.get_height()
        if math.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_summary_bars(runs: List[ParsedRun], output_path: Path) -> None:
    labels = [run.label for run in runs]
    x = list(range(len(runs)))
    losses = [run.final_loss if run.final_loss is not None else float("nan") for run in runs]
    throughputs = [
        run.mean_tokens_per_sec_gpu_after_warmup
        if run.mean_tokens_per_sec_gpu_after_warmup is not None
        else float("nan")
        for run in runs
    ]
    colors = [style_for_run(run)["color"] for run in runs]
    hatches = ["", "", "//", "//"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    bars_loss = axes[0].bar(x, losses, color=colors, alpha=0.9)
    for bar, hatch in zip(bars_loss, hatches):
        bar.set_hatch(hatch)
    axes[0].set_title("Quick Run Summary Metrics")
    axes[0].set_ylabel("Final LM Loss")
    axes[0].grid(True, axis="y", alpha=0.25)
    _bar_label(axes[0], bars_loss, "{:.3f}")

    bars_tput = axes[1].bar(x, throughputs, color=colors, alpha=0.9)
    for bar, hatch in zip(bars_tput, hatches):
        bar.set_hatch(hatch)
    axes[1].set_ylabel("Mean Tokens/s/GPU\n(iters 21-200)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)
    _bar_label(axes[1], bars_tput, "{:.0f}")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    primary_runs: List[ParsedRun],
    candidate_runs: List[ParsedRun],
    output_path: Path,
    plot_paths: List[Path],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_runs = [
        run for run in candidate_runs if run.path.name in PREFERRED_FAILED_LOGS
    ]
    superseded_runs = [
        run for run in candidate_runs if run.path.name in SUPERSEDED_SUCCESSFUL_LOGS
    ]

    lines: List[str] = []
    lines.append("# Quick Results Summary")
    lines.append("")
    lines.append("## Setup")
    lines.append("- Dataset: `climbmix_small`")
    lines.append("- Tokenizer: GPT-2 BPE")
    lines.append("- Hardware: 1 GPU quick smoke runs on Clariden")
    lines.append("- Model scale: tiny 6L / 512H / 1408F setup, about 68M-71M params")
    lines.append("- Training length: 200 iterations per plotted run")
    lines.append("")
    lines.append("## Plots")
    for path in plot_paths:
        rel = path.relative_to(REPO_ROOT)
        lines.append(f"- `{rel}`")
    lines.append("")
    lines.append("## Primary Logs Used")
    for run in primary_runs:
        rel = run.path
        lines.append(f"- `{rel}`")
    lines.append("")
    lines.append("## Final Metrics")
    lines.append("")
    lines.append("| Run | Job ID | Status | Final loss | Mean tok/s/GPU (iters 21-200) | Final tok/s/GPU | Params | Max reserved MB | |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for run in primary_runs:
        lines.append(
            "| "
            + f"{run.label} | {run.job_id} | {run.status_text()} | "
            + f"{run.final_loss:.6f} | "
            + f"{run.mean_tokens_per_sec_gpu_after_warmup:.1f} | "
            + f"{run.final_tokens_per_sec_gpu:.1f} | "
            + f"{run.params or 0} | "
            + f"{run.max_reserved_mb:.2f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- These are short engineering smokes, not long-horizon training runs.")
    lines.append("- Final loss and throughput are useful for quick bring-up comparisons, but not yet publication-grade evidence.")
    lines.append("- Mean tokens/sec/GPU uses iterations 21-200 to reduce warmup distortion.")
    lines.append("")

    if superseded_runs:
        lines.append("## Superseded Successful Runs")
        for run in superseded_runs:
            desc = SUPERSEDED_SUCCESSFUL_LOGS[run.path.name]
            lines.append(
                f"- `{run.path}`: {desc} Final loss `{run.final_loss:.6f}`, "
                f"final tokens/sec/GPU `{run.final_tokens_per_sec_gpu:.1f}`."
            )
        lines.append("")

    if failed_runs:
        lines.append("## Failed Or Excluded Runs")
        for run in failed_runs:
            desc = PREFERRED_FAILED_LOGS[run.path.name]
            lines.append(f"- `{run.path}`: {desc}")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_tracker_section(summary_path: Path, plot_paths: List[Path]) -> None:
    existing = TRACKER_MD.read_text(encoding="utf-8")
    marker = "## Quick Results Plots"
    if marker in existing:
        return

    rel_summary = summary_path.relative_to(REPO_ROOT)
    rel_plots = ", ".join(f"`{path.relative_to(REPO_ROOT)}`" for path in plot_paths)
    addition = (
        "\n## Quick Results Plots\n"
        f"- Parsed local quick-run logs and generated summary plots under {rel_plots}.\n"
        f"- Detailed metrics and run notes are recorded in `{rel_summary}`.\n"
    )
    TRACKER_MD.write_text(existing.rstrip() + "\n" + addition, encoding="utf-8")


def print_metric_table(primary_runs: List[ParsedRun]) -> None:
    header = (
        f"{'Run':<16} {'Job':>8} {'Status':>10} {'Final loss':>12} "
        f"{'Mean tok/s':>12} {'Final tok/s':>12}"
    )
    print(header)
    print("-" * len(header))
    for run in primary_runs:
        print(
            f"{run.label:<16} {run.job_id:>8} {run.status_text():>10} "
            f"{run.final_loss:>12.6f} {run.mean_tokens_per_sec_gpu_after_warmup:>12.1f} "
            f"{run.final_tokens_per_sec_gpu:>12.1f}"
        )


def main() -> None:
    configure_matplotlib()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    candidate_paths = discover_candidate_logs(RUNS_DIR)
    candidate_runs = parse_candidates(candidate_paths)

    primary_runs = [
        parse_run(
            path=Path(spec["path"]),
            label=spec["label"],
            architecture=spec["architecture"],
            optimizer_family=spec["optimizer"],
        )
        for spec in PRIMARY_MANIFEST
    ]

    plot_paths = [
        PLOTS_DIR / "attention_loss_curves.png",
        PLOTS_DIR / "optimizer_loss_curves.png",
        PLOTS_DIR / "summary_bars.png",
    ]

    plot_attention_loss_curves(primary_runs, plot_paths[0])
    plot_optimizer_loss_curves(primary_runs, plot_paths[1])
    plot_summary_bars(primary_runs, plot_paths[2])
    write_summary(primary_runs, candidate_runs, SUMMARY_MD, plot_paths)
    append_tracker_section(SUMMARY_MD, plot_paths)

    print("Generated files:")
    for path in plot_paths + [SUMMARY_MD]:
        print(path)
    print()
    print_metric_table(primary_runs)


if __name__ == "__main__":
    main()
