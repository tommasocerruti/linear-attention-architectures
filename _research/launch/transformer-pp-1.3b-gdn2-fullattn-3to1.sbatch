#!/bin/bash
#
# Hybrid Gated DeltaNet2 ~1.3B on 1 GH200 node (4 GPUs).
# 24L / 2048H / 5632F / 32h / 8kv (GQA), with a 3:1
# GDN2:standard full self-attention layer pattern.
# AdamW-style Adam, WSD, bf16. Defaults to 60B FineWeb-Edu tokens.
#
#SBATCH --job-name=transformer-pp-1.3b-gdn2-fullattn-3to1
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=72:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail

CODE_DIR=${CODE_DIR:-/users/$USER/cler}
SCRIPT_DIR=$CODE_DIR/_research/launch
if [ ! -f "$CODE_DIR/pretrain_gpt.py" ] || [ ! -d "$CODE_DIR/megatron" ]; then
    echo "ERROR: CLER code checkout not found at $CODE_DIR" >&2
    exit 1
fi
cd "$CODE_DIR"

RUN_ID=transformer-pp-1.3b-gdn2-fullattn-3to1
GIT_SHA=$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)

export NUM_LAYERS=${NUM_LAYERS:-24}
export HIDDEN=${HIDDEN:-2048}
export FFN_HIDDEN=${FFN_HIDDEN:-5632}
export NUM_HEADS=${NUM_HEADS:-32}
export NUM_KV_HEADS=${NUM_KV_HEADS:-8}
export LINEAR_ATTENTION_FREQ=${LINEAR_ATTENTION_FREQ:-4}
export LINEAR_NUM_KEY_HEADS=${LINEAR_NUM_KEY_HEADS:-8}
export LINEAR_NUM_VALUE_HEADS=${LINEAR_NUM_VALUE_HEADS:-16}
export LINEAR_KEY_HEAD_DIM=${LINEAR_KEY_HEAD_DIM:-64}
export LINEAR_VALUE_HEAD_DIM=${LINEAR_VALUE_HEAD_DIM:-64}

export MBS=${MBS:-2}
export GBS=${GBS:-128}
export SEQ_LEN=${SEQ_LEN:-4096}
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-14648438}
export EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-4310}
export LR=${LR:-2e-4}
export MIN_LR=${MIN_LR:-2e-5}
export LR_WSD_DECAY_SAMPLES=${LR_WSD_DECAY_SAMPLES:-2929688}
export RECOMPUTE=${RECOMPUTE:-1}
export ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT:-1}
export EVAL_INTERVAL=${EVAL_INTERVAL:-$TRAIN_SAMPLES}
export EVAL_ITERS=${EVAL_ITERS:-0}
export WANDB_MODEL_SIZE=${WANDB_MODEL_SIZE:-1.3B}
export EXP_NAME=${EXP_NAME:-$RUN_ID-$GIT_SHA}
export APERTUS_TRACK=${APERTUS_TRACK:-$RUN_ID}

exec bash "$SCRIPT_DIR/transformer-pp-350m-gdn2.sbatch"
