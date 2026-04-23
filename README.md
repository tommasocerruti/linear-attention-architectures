# Cross-Layer Residual Error Routing (CLER)

CLER is our project on cross-layer residual error routing for large language
model training on the Swiss AI Alps / Clariden cluster. The initial proposal
is available at [tracker/week0/proposal.md](tracker/week0/proposal.md).

This repository is built on top of
[ischlag/megatron-lm-research-baseline](https://github.com/ischlag/megatron-lm-research-baseline),
which itself builds on
[NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM). We use that
baseline as the training and experimentation framework, and add our
project-specific tracking, experiments, and modifications here.

The first CLER baseline track uses
[FineWeb-Edu 100BT shuffled](https://huggingface.co/datasets/HuggingFaceFW/fineweb_edu_100BT-shuffled)
15B-token prefixes converted to Megatron binary format with a LLaMA-2
SentencePiece tokenizer. The inherited GPT-2/ClimbMix path remains useful for
quick pipeline smoke tests, but reported CLER baselines should stay on LLaMA-2
for paper comparability.


## Project overview

The goal of this repository is to support reproducible CLER experiments in a
Megatron-LM-based training stack tuned for Clariden (GH200 Grace-Hopper nodes,
4 GPUs per node, Slingshot-11 interconnect).

At a high level, this repository serves two purposes:

1. Provide a stable research baseline for launching and comparing training runs.
2. Host CLER-specific experiment tracking, notes, and future implementation work.

## Git workflow for this project

- `git push` pushes our work to our private repository on `origin`.
- `git pull` pulls the latest changes from the tracked branch on `origin`.
- `git pull upstream main` pulls updates from the professor repository.
- After pulling upstream changes, run `git submodule update --init --recursive`.

## Project tracker

Project planning and weekly tracking live under [`tracker/`](tracker/README.md).

- [`tracker/README.md`](tracker/README.md) contains the week-by-week schedule.
- [`tracker/tom.md`](tracker/tom.md), [`tracker/tim.md`](tracker/tim.md),
  [`tracker/ling.md`](tracker/ling.md), and [`tracker/george.md`](tracker/george.md)
  are the individual diary files.

## Repository base

The underlying training stack comes from the Megatron-LM research baseline for
reproducible comparisons of dense and MoE language models on Clariden.

The sections below document that baseline environment, launch flow, and
configurations, which CLER builds on rather than replaces.

## Quick start

Intended environment: Clariden (Swiss AI Alps, GH200 nodes, SLURM +
enroot container runtime). Everything below assumes that cluster; to
adapt to a different site, swap the EDF and dataset path and rewrite
the `#SBATCH` headers to fit your scheduler.

**First-time setup**: add to your `~/.bashrc` (or source once per
shell) so `sbatch` picks up sane defaults and the sbatches can find
your dataset:

```bash
export SBATCH_ACCOUNT=<your-slurm-account>        # required
export SBATCH_RESERVATION=<your-reservation>      # optional; omit for normal queue
export WANDB_API_KEY=<your-wandb-key>             # optional; omit to log locally only
export WANDB_PROJECT=<your-wandb-project>         # optional; defaults to megatron-lm-research-baseline
export LLAMA2_TOKENIZER_MODEL=/path/to/tokenizer.model # required for FineWeb-Edu conversion/training
export MEGATRON_DATA_PATH=/path/to/fineweb_edu_15b_llama2_text_document # required; Megatron-binary prefix without .bin/.idx
```

Build the FineWeb-Edu 15B prefix once from the repo root:

```bash
sbatch _research/data/convert_fineweb_edu.sbatch
```

The conversion job uses `LLAMA2_TOKENIZER_MODEL` for token counting and
Megatron preprocessing, then prints the `MEGATRON_DATA_PATH` value when it
completes. Details and the 100B variant are documented in
[`_research/data/README.md`](_research/data/README.md).

For smoke tests before FineWeb-Edu conversion finishes, a tokenized ClimbMix
copy is available on Clariden at a shared read-only path:

```bash
export MEGATRON_DATA_PATH=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
```

`sbatch` reads `SBATCH_ACCOUNT` / `SBATCH_RESERVATION` natively, so no
wrapper is needed. The sbatches abort with a clear error at submit time
if `MEGATRON_DATA_PATH` is unset.

**Running a job:**

```bash
# Clone into the expected path. The sbatches write the Python package
# dir, caches, and SLURM logs to this exact location; cloning elsewhere
# will scatter outputs across two directories.
cd /iopsstor/scratch/cscs/$USER
git clone git@github.com:tommasocerruti/cler.git
cd cler

# Submit. The alps3 enroot container +
# _research/launch/install_python_deps.sh handle the Python environment
# inside the job; no local install required.
sbatch _research/launch/transformer-pp-350m-adamw-smoke.sbatch
# if the smoke run is clean, launch the full AdamW baseline:
sbatch _research/launch/transformer-pp-350m-adamw.sbatch
# or the Gated DeltaNet hybrid baseline:
sbatch _research/launch/transformer-pp-350m-gdn.sbatch
# or the NorMuon variant:
sbatch _research/launch/transformer-pp-350m-muon.sbatch
# or a 1B-token quick reference (AdamW, ~30 min — good first smoke test):
sbatch _research/launch/transformer-pp-350m-ablation.sbatch
```

To try a variant (different optimizer, LR, schedule, etc.), copy an
existing sbatch and edit it. Frozen ablation runs live under
`_research/leaderboards/<size>/runs/`.

Python dependencies (`transformers`, `datasets`, `sentencepiece`, `wandb`,
`flash-linear-attention`, `emerging-optimizers`) are
installed into `_research/packages/` inside the container on first run
via `_research/launch/install_python_deps.sh`; no `pip install` on the
login node is needed.

See [README.nvidia.md](README.nvidia.md) for the original NVIDIA Megatron-LM
documentation.

**AI assistants**: read [AGENTS.md](AGENTS.md) before making changes. It
documents the repo layout, experiment flow, and conventions.

## Leaderboards

Ranked run lists per model size; each entry is a self-contained sbatch + W&B link.

- [`350m-ablation`](_research/leaderboards/350m-ablation/README.md) — 1B-token optimizer ablations (NorMuon / Muon / AdamW)
- [`350m`](_research/leaderboards/350m/README.md) — 15B-token full baseline (placeholder)
- [`760m`](_research/leaderboards/760m/README.md) — 30B-token full baseline (placeholder)
- [`1.3b`](_research/leaderboards/1.3b/README.md) — 100B-token full baseline (placeholder)
- [`2.7b`](_research/leaderboards/2.7b/README.md) — 300B-token full baseline (placeholder)

## Configurations

Transformer++ baselines (SwiGLU, RMSNorm, RoPE, GQA, AdamW, WSD schedule,
bf16) on the configured Megatron dataset. The current CLER baselines use
FineWeb-Edu with LLaMA-2 tokenization; legacy ClimbMix smoke tests use GPT-2
BPE. All configs use GBS=128
sequences (524K tokens/step) and are tuned for GH200 nodes with 4 GPUs each.

| config | params | tokens | nodes | GPUs | DP | MBS | est. wall | GPU-h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transformer-pp-350m-ablation` | 350M | 1B | 1 | 4 | 4 | 16 | ~30 min | 2 |
| `transformer-pp-350m-{adamw,muon}` | 350M | 15B | 1 | 4 | 4 | 16 | ~8 h | 31 |
| `transformer-pp-760m-adamw` | 760M | 30B | 4 | 16 | 16 | 4 | ~7 h | 115 |
| `transformer-pp-1.3b-adamw` | 1.3B | 100B | 8 | 32 | 32 | 2 | ~19 h | 596 |
| `transformer-pp-2.7b-adamw` | 2.7B | 300B | 16 | 64 | 64 | 1 | ~60 h | 3,834 |

The `-muon` variant at 350M uses NorMuon (`adaptive_muon` + `normuon`) with
matrix LR 3.6e-4 and scalar LR 1.5e-3; it differs from `-adamw` only in
the optimizer block (validated on `-ablation`: NorMuon beats AdamW by
~0.10 nats final loss).

### Architectures

| | 350M | 760M | 1.3B | 2.7B |
| --- | ---: | ---: | ---: | ---: |
| hidden | 1024 | 1536 | 2048 | 2560 |
| layers | 24 | 24 | 24 | 32 |
| heads / kv_heads | 16 / 4 | 24 / 8 | 32 / 8 | 32 / 8 |
| ffn (SwiGLU) | 2560 | 4096 | 5632 | 7680 |
| peak LR | 3e-4 | 2.5e-4 | 2e-4 | 1.6e-4 |

### Launch vs leaderboards

| folder | purpose |
| --- | --- |
| `_research/launch/` | launchable sbatches: baseline full-run per size (`-adamw`, `-muon`) plus a short `-ablation` 1B-token reference. All hparams pinned. |
| `_research/leaderboards/` | historical ranked runs. Each entry is a frozen, reproducible sbatch + W&B link. |

Workflow for a new ablation: copy an existing sbatch, edit the optimizer,
LR, or schedule, run it, and if it wins, snapshot the file into
`_research/leaderboards/<size>/runs/` and add a row to that size's `README.md`.

## Baseline features inherited from the professor repository

The current repository inherits several research-oriented extensions from
the professor baseline.

### Logging patch (`_research/logging_patch/`)

A monkey-patch layer that hooks into Megatron's `training_log` and
`setup_model_and_optimizer` without modifying upstream source files. All
configuration is via environment variables; the patch is activated by a
two-line import in `pretrain_gpt.py`.

Per-step metrics are written to an append-only JSONL log (`<run>.jsonl`)
plus a metadata sidecar (`<run>.meta.json`):

| metric | env var | default |
| --- | --- | --- |
| `train_loss`, `lr`, `grad_norm`, `params_norm`, `tput` | always on | -- |
| MFU (model FLOPs utilization) | always on | -- |
| Per-layer activation norms and max | `APERTUS_LOG_ACT_STATS` | on |
| Top-1 next-token accuracy (TP-aware) | `APERTUS_LOG_TOP1_ACC` | on |
| Per-parameter gradient norms | `APERTUS_LOG_PER_LAYER_GRADS` | off |
| Loss spike detection (rolling z-score) | `APERTUS_LOG_LOSS_SPIKES` | off |
| Startup phase timeline (sbatch, srun, container, dist init, model build, first iters) | always on | -- |

The JSONL writer scales O(1) per step (no full-file rewrite). An analysis
loader at `_research/analyse/load_runs.py` reads both the new JSONL format and
legacy single-file JSON for backward compatibility.

### Muon / NorMuon (`--optimizer muon`, `--optimizer adaptive_muon`)

Muon uses Newton-Schulz orthogonalization on the momentum matrix to take
spectrally-normalized steps. Routing: 2D matrix params use Muon; scalar
params (embeddings, norms, biases) use AdamW via the upstream
`_is_nonlinear_or_embedding` predicate. Relevant flags added on top of
upstream include:

- `--muon-scalar-lr` decouples the AdamW-group LR from the Muon-group LR.
- `--muon-scalar-weight-decay` can keep weight decay on the Muon matrix
  group only.
- `--adaptive-muon-moment2-method` selects `adamuon` or `normuon`.

**Do not use `--overlap-param-gather` when training with Muon.**
This is safe for AdamW because Adam's update is elementwise, but Muon's
update depends on the full momentum matrix. If `--overlap-param-gather`
assembles the matrix from shards gathered at inconsistent async states,
Newton-Schulz sees a corrupted matrix and the update spectrum is wrong.
Empirically this shows up as monotonically growing `params_norm` and
`grad_norm` from step 1. Keep `--overlap-grad-reduce` and
`--use-distributed-optimizer`; both were verified safe for Muon.

### AdEMAMix optimizer (`--optimizer ademamix`)

Ported from the
[swiss-ai/Megatron-LM](https://github.com/swiss-ai/Megatron-LM) fork.
AdEMAMix adds a slow EMA on top of Adam's fast EMA for better convergence
at scale (Pagliardini et al., 2025). Relevant flags:

- `--ademamix-alpha` (default 2.0)
- `--ademamix-beta3` (default 0.9999)
- `--ademamix-beta3-warmup` (warmup steps for beta3 in half-life space)
- `--ademamix-alpha-warmup` (warmup steps for alpha)

### xIELU activation (`--xielu`)

Learnable per-layer activation from the Apertus architecture (two parameters
per layer: `alpha_p`, `alpha_n`). This is a non-gated 2-matrix MLP
(up projection, activation, down projection). Use it with
`--ffn-hidden-size` set to the full intermediate dimension.

### Goldfish loss (`--goldfish-loss`)

Token-level memorization suppression (Hans et al., 2024). It masks ~1/k of
target labels using a deterministic hash of the preceding h tokens.
Relevant flags:

- `--goldfish-k` (drop fraction, default 50)
- `--goldfish-h` (context width for hashing, default 50)

### Determinism (`APERTUS_DETERMINISTIC=1`)

Optional flag that enables `torch.use_deterministic_algorithms`,
deterministic cuDNN, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` for bitwise
reproducibility, at a compute cost.

## Syncing with the professor baseline

To bring the latest changes from the professor repository into your local
`main` branch:

```bash
git pull upstream main
git submodule update --init --recursive
```

The `upstream` remote points to
[`ischlag/megatron-lm-research-baseline`](https://github.com/ischlag/megatron-lm-research-baseline).
Your own work should be pushed to `origin`, which points to your private CLER
repository.
