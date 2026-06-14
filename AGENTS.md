# Agents Guide

Technical orientation for AI assistants working in this repo. Read this
once on first entry; use it as a reference, not a script.

## What this is

The private repository for **Cross-Layer Residual Error Routing (CLER)**,
built on top of
[ischlag/megatron-lm-research-baseline](https://github.com/ischlag/megatron-lm-research-baseline),
which itself builds on [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

Treat the professor baseline as the training scaffold and core codebase.
Treat CLER-specific tracking, notes, and future implementation work as the
project layer on top.

## Repo layout

| path | contents |
| --- | --- |
| `_research/launch/` | launchable sbatches: baselines (`transformer-pp-<size>-<adamw\|muon>.sbatch`) and a 1B-token quick reference (`-ablation.sbatch`) |
| `_research/leaderboards/<size>/README.md` | ranked result table + W&B links for that size |
| `_research/leaderboards/<size>/runs/NN-*.sbatch` | frozen, self-contained sbatches |
| `_research/logging_patch/` | JSONL + wandb telemetry, activated by a two-line hook in `pretrain_gpt.py` |
| `_research/data/` | tokenizer files (GPT-2 BPE) |
| `_research/results/` | gitignored run outputs (logs, tensorboard, wandb) |
| `megatron/core/optimizer/emerging_optimizers.py` | Muon / AdaMuon / NorMuon glue |
| `megatron/core/optimizer/ademamix.py` | AdEMAMix port |
| `megatron/core/activations.py` | xIELU activation |
| `megatron/training/arguments.py` | CLI flags (baseline + local additions) |
| `megatron/core/optimizer/optimizer_config.py` | `OptimizerConfig` fields; mirror each new CLI flag here |
| `pretrain_gpt.py` | entrypoint + logging-patch hook |

Everything not listed is baseline Megatron or baseline research infra. Treat
it as read-only unless you're making a targeted, justified change.

## How experiments flow

```text
copy an existing sbatch in _research/launch/, edit the optimizer/LR/schedule block
        ↓  if a config wins
snapshot a self-contained sbatch into
_research/leaderboards/<size>/runs/NN-*.sbatch
and add a row to that README
```

## Good practices

- **Keep CLER framing explicit** in docs: this is a CLER repo built on a baseline.
- **Prefer adding project docs under `docs/`** rather than cluttering the repo root.
- **Flag-gate everything new**, default off. Add the CLI flag in
  `arguments.py` and the field in `optimizer_config.py`; don't change
  existing default behaviour.
- **Don't remove the logging-patch hook** in `pretrain_gpt.py`.
- **Muon + `--overlap-param-gather` is broken**. Keep
  `--use-distributed-optimizer` and `--overlap-grad-reduce`, but drop
  `--overlap-param-gather` for `--optimizer muon|adaptive_muon`.
- **No SLURM job without a stopping condition** (`--train-iters`,
  `--train-samples`, or `--exit-duration-in-mins`).
- **Cluster specifics are hardcoded** in `#SBATCH` headers. To port to a
  new site, edit the header block and `--data-path`.
- **Do not modify unrelated baseline code** just to make the repo feel more CLER-specific.

## Git workflow

- `origin` = private CLER repository
- `upstream` = professor baseline repository

Useful commands:

```bash
git push
git pull
git pull upstream main
git submodule update --init --recursive
```

## When to pause and ask the human

- Adding cluster-wide or repo-wide dependencies.
- Launching jobs above the node cap the human has set.
- Changing remotes, branch strategy, or repo structure.
- Touching upstream source outside the flag-gated exceptions.
- Deleting results, branches, or worktrees you didn't create.
