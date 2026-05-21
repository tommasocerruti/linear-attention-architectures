# CLER v1: PyTorch Gated DeltaNet

CLER v1 implements Cross-Layer Residual Error Routing for the pure PyTorch
Gated DeltaNet attention variant. It is intentionally not kernelized yet.

## What It Does

CLER routes the Gated DeltaNet delta-rule write residual from one GDN layer to
the next GDN layer:

```text
value_l = value_l + gamma_l * residual_{l-1}
```

The routed signal is the `v_new` tensor inside the PyTorch gated delta rule.
For v1, softmax/SDPA layers neither consume nor emit CLER residuals. In a
hybrid pattern such as `--linear-attention-freq 3`, CLER routes across adjacent
PyTorch GDN layers and resets at SDPA layers.

Important limits:

- supported only with `--experimental-attention-variant gated_delta_net_pytorch`
- default-off via `--cler-enabled`
- no cross-pipeline-stage CLER state in v1
- no fused `gated_delta_net.py` or FLA kernel changes yet
- no value-dimension projection in v1; the GDN layers must share the same value
  head layout, as the current launchers do

## Code Pointers

- `megatron/core/ssm/gated_delta_net_pytorch.py`
  - captures `v_new` when `return_residual=True`
  - injects `cler_gamma * cler_residual` into the current layer value tensor
- `megatron/core/transformer/transformer_layer.py`
  - passes CLER residual into supported self-attention modules
  - exposes the current layer's CLER residual after self-attention
- `megatron/core/transformer/transformer_block.py`
  - carries the residual from layer to layer
  - resets residual state across non-CLER layers
- `megatron/core/transformer/transformer_config.py`
  - adds `cler_enabled`, `cler_gamma_init`, `cler_detach_residual`
  - rejects unsupported CLER variants
- `megatron/core/models/gpt/experimental_attention_variant_module_specs.py`
  - treats `gated_delta_net_pytorch` as a linear-attention variant, so
    `--linear-attention-freq 3` means GDN/GDN/SDPA.

## CLI Flags

```bash
--experimental-attention-variant gated_delta_net_pytorch
--linear-attention-freq 3
--cler-enabled
--cler-gamma-init 0.0
```

Optional:

```bash
--cler-detach-residual
```

`--cler-gamma-init 0.0` is the recommended default. It makes CLER-on initialize
from baseline behavior while still learning a per-layer scalar `gamma_l`.

## Tests

Use the active project Python environment rather than a hardcoded local conda
name:

```bash
python tests/unit_tests/ssm/test_gated_delta_net_pytorch_cler.py
```

If the environment has pytest:

```bash
python -m pytest tests/unit_tests/ssm/test_gated_delta_net_pytorch_cler.py
```

These tests are CPU/direct-run friendly and cover:

- old two-output delta-rule behavior
- residual tensor shape and dtype
- single-token residual equals the value target with zero initial state
- gamma-zero/gamma-one value injection behavior
- config validation and linear-attention pattern selection

## Launch On Alps

Prerequisites:

```bash
export MEGATRON_DATA_PATH=/path/to/fineweb_edu_15b_llama2_text_document
export LLAMA2_TOKENIZER_MODEL=/path/to/tokenizer.model
```

Canonical 350M CLER v1 run:

```bash
sbatch _research/launch/transformer-pp-350m-cler-v1.sbatch
```

Short smoke using the same launcher:

```bash
TRAIN_ITERS=50 EXIT_DURATION_IN_MINS=25 EVAL_INTERVAL=25 EVAL_ITERS=1 \
  sbatch --partition=debug --time=00:45:00 \
  _research/launch/transformer-pp-350m-cler-v1.sbatch
```

Experimental 1.3B scale target:

```bash
export MEGATRON_DATA_PATH=/path/to/fineweb_edu_100b_llama2_text_document
export LLAMA2_TOKENIZER_MODEL=/path/to/tokenizer.model
sbatch _research/launch/transformer-pp-1.3b-cler-v1.sbatch
```

Run the 350M smoke and 350M full run before the 1.3B launcher. The 1.3B script
uses the repo's 1.3B Transformer++ shape with PyTorch GDN/CLER dimensions scaled
to 8 key heads and 16 value heads.

## Canonical 350M Sbatch

The committed launcher is
`_research/launch/transformer-pp-350m-cler-v1.sbatch`:

```bash
#!/bin/bash
#
# CLER v1 PyTorch Gated DeltaNet ~350M run on 1 GH200 node (4 GPUs).
# Uses the FineWeb-Edu/LLaMA-2 PyTorch GDN launcher with CLER enabled.
#
# Required environment:
#   MEGATRON_DATA_PATH=/path/to/fineweb_edu_15b_llama2_text_document
#   LLAMA2_TOKENIZER_MODEL=/path/to/tokenizer.model
#
# Optional smoke override:
#   TRAIN_ITERS=50 EXIT_DURATION_IN_MINS=25 EVAL_INTERVAL=25 EVAL_ITERS=1 \
#     sbatch --partition=debug --time=00:45:00 _research/launch/transformer-pp-350m-cler-v1.sbatch
#
#SBATCH --job-name=transformer-pp-350m-cler-v1
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=10:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail

REPO_ROOT=/iopsstor/scratch/cscs/$USER/cler
SCRIPT_DIR=$REPO_ROOT/_research/launch
cd "$REPO_ROOT"

GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)

export CLER_ENABLED=1
export CLER_GAMMA_INIT=${CLER_GAMMA_INIT:-0.0}
export CLER_DETACH_RESIDUAL=${CLER_DETACH_RESIDUAL:-0}

export NUM_LAYERS=${NUM_LAYERS:-20}
export HIDDEN=${HIDDEN:-1024}
export FFN_HIDDEN=${FFN_HIDDEN:-2816}
export NUM_HEADS=${NUM_HEADS:-16}
export NUM_KV_HEADS=${NUM_KV_HEADS:-4}
export MBS=${MBS:-16}
export GBS=${GBS:-128}
export SEQ_LEN=${SEQ_LEN:-4096}
export LINEAR_ATTENTION_FREQ=${LINEAR_ATTENTION_FREQ:-3}
export LINEAR_KEY_HEAD_DIM=${LINEAR_KEY_HEAD_DIM:-64}
export LINEAR_VALUE_HEAD_DIM=${LINEAR_VALUE_HEAD_DIM:-64}
export LINEAR_NUM_KEY_HEADS=${LINEAR_NUM_KEY_HEADS:-4}
export LINEAR_NUM_VALUE_HEADS=${LINEAR_NUM_VALUE_HEADS:-8}

export EXP_NAME=${EXP_NAME:-transformer-pp-350m-cler-v1-$GIT_SHA}
export APERTUS_FEATURE=${APERTUS_FEATURE:-cler-v1-gdn-pytorch}
export APERTUS_TRACK=${APERTUS_TRACK:-transformer-pp-350m-cler-v1}
export WANDB_ARCH=${WANDB_ARCH:-cler-v1-gdn}
export WANDB_MODEL_SIZE=${WANDB_MODEL_SIZE:-350M}

exec bash "$SCRIPT_DIR/transformer-pp-350m-gdn-pytorch.sbatch"
```
