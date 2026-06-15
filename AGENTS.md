# Agents Guide

Technical orientation for AI assistants working in this repository. Treat it as the release repo for the technical report "Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing", not as a tracker-first CLER workspace.

## What This Is

This is a focused Megatron-LM fork for reproducing the paper's linear-attention experiments. It keeps the Megatron core source, tests, tools, and package metadata available for backward compatibility, while adding the paper mechanisms and Clariden launch/evaluation scripts needed for reproduction.

The supported paper mechanisms are DeltaNet, Gated DeltaNet, Kimi Delta Attention, Gated DeltaNet-2, CLER, and CLVR. The experiment launchers under `_research/launch/` are the executable source of truth for run-specific flags.

## Repo Layout

| Path | Contents |
| --- | --- |
| `megatron/` | Megatron core plus local linear-attention and routing implementations |
| `_research/data/` | FineWeb-Edu conversion scripts and tokenizer assets |
| `_research/launch/` | Final launchers, smoke checks, scale runs, and ablation submitters |
| `_research/eval/` | `lm-eval` wrappers for native Megatron checkpoints |
| `tools/run_loglikelihood_scoring_server.py` | Full-sequence scoring server used for downstream evaluation |
| `docs/reproducibility.md` | Longer reproduction notes |
| `tests/unit_tests/ssm/` | Focused tests for DeltaNet/GDN/CLER routing behavior |

Everything else is inherited Megatron or baseline infrastructure. Treat broad core rewrites as high risk.

## Working Rules

- Preserve backward compatibility unless the user explicitly asks for a breaking paper-release change.
- Keep paper reproduction files: final 350M/15B launchers, shared `transformer-pp-350m-linear-muon.sbatch`, smoke launchers, LR and sequence-scaling submitters, larger-scale routing launchers, `_research/data`, `_research/eval`, tests, tools, and Megatron source.
- Do not recreate tracker-style planning folders. Put durable docs in `docs/` or in the relevant `_research/*/README.md`.
- Flag-gate new behavior and keep defaults compatible with inherited Megatron behavior.
- Do not remove the logging-patch hook in `pretrain_gpt.py`.
- Muon with `--overlap-param-gather` is broken here. Keep `--use-distributed-optimizer` and `--overlap-grad-reduce`, but do not add `--overlap-param-gather` for `--optimizer muon|adaptive_muon`.
- Never launch a SLURM job without a stopping condition such as `--train-iters`, `--train-samples`, or `--exit-duration-in-mins`.
- Cluster paths in launchers are Clariden-specific. To port the repo, edit `#SBATCH` headers, scratch paths, container environment, and data/tokenizer variables explicitly.

## Verification Habits

For code changes, prefer focused tests first:

```bash
python3 -m pytest tests/unit_tests/ssm/test_cler_delta_net_pytorch.py tests/unit_tests/ssm/test_gated_delta_net_pytorch_cler.py tests/unit_tests/ssm/test_cler_fast_rules.py -q
```

For release cleanup changes, also run AST parsing over tracked Python files, `bash -n` over tracked shell and sbatch files, and `rg` checks for stale tracker or old-repo references.

## When To Pause

Ask the human before adding repo-wide dependencies, changing remotes or branch strategy, launching expensive jobs, deleting results/checkpoints, or modifying broad inherited Megatron behavior unrelated to the paper mechanisms.
