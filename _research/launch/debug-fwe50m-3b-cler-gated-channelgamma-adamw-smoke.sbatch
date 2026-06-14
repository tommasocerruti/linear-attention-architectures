#!/bin/bash
#
# Debug smoke for the 3B-ish CLER-Gated per-channel AdamW scale probe.
#
# Submit before the full large job:
#   sbatch _research/launch/debug-fwe50m-3b-cler-gated-channelgamma-adamw-smoke.sbatch
#
# If running from another account/checkout, override:
#   CLER_REPO_ROOT=/path/to/cler
#   MEGATRON_DATA_PATH=/path/to/fineweb_edu_50m_llama2_smoke_text_document
#   LLAMA2_TOKENIZER_MODEL=/path/to/tokenizer.model
#
#SBATCH --job-name=dbg-3b-cler-cg-a
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:59:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail

REPO_ROOT_DEFAULT=${SLURM_SUBMIT_DIR:-/users/course_00252/cler}
export CLER_REPO_ROOT=${CLER_REPO_ROOT:-$REPO_ROOT_DEFAULT}

WANDB_ENV_FILE=${WANDB_ENV_FILE:-$HOME/.wandb_cler_env}
HF_ENV_FILE=${HF_ENV_FILE:-$HOME/.hf_cler_env}
if [ -f "$WANDB_ENV_FILE" ]; then
    source "$WANDB_ENV_FILE"
fi
if [ -f "$HF_ENV_FILE" ]; then
    source "$HF_ENV_FILE"
fi

export MEGATRON_DATA_PATH=${MEGATRON_DATA_PATH:-/iopsstor/scratch/cscs/course_00252/cler/_research/results/data/fineweb_edu/fineweb_edu_50m_llama2_smoke/fineweb_edu_50m_llama2_smoke_text_document}
export LLAMA2_TOKENIZER_MODEL=${LLAMA2_TOKENIZER_MODEL:-/iopsstor/scratch/cscs/course_00252/llama2-tokenizer/tokenizer.model}
export WANDB_ENTITY=${WANDB_ENTITY:-cler}
export WANDB_PROJECT=${WANDB_PROJECT:-clerv1-runs}
export WANDB_GROUP=${WANDB_GROUP:-debug-fwe50m-3b-scale-probe-20260518}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=fineweb_edu_50m_llama2_smoke
export WANDB_IS_SMOKE=1
export WANDB_MODEL_SIZE=3B
export WANDB_ARCH=cler-v1-gdn-pytorch-channelgamma
export WANDB_KERNEL=torch
export APERTUS_FEATURE=debug-3b-cler-gated-channelgamma-adamw
export APERTUS_TRACK=debug-fwe50m-3b-cler-gated-channelgamma-adamw
export EXP_NAME=debug-3b-llama2-fwe50m-cler-gated-channelgamma-adamw-gamma1e-2-compile

export NUM_LAYERS=32
export HIDDEN=2560
export FFN_HIDDEN=7680
export NUM_HEADS=32
export NUM_KV_HEADS=8
export MBS=1
export GBS=4
export SEQ_LEN=4096
export TRAIN_ITERS=2
export LR_WARMUP_ITERS=1
export EVAL_INTERVAL=2
export EVAL_ITERS=0
export EXIT_DURATION_IN_MINS=50
export LR=1.6e-4
export MIN_LR=1.6e-5
export TP=${TP:-1}
export PP=${PP:-1}
export RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-32}
export LINEAR_ATTENTION_FREQ=3
export LINEAR_KEY_HEAD_DIM=64
export LINEAR_VALUE_HEAD_DIM=64
export LINEAR_NUM_KEY_HEADS=8
export LINEAR_NUM_VALUE_HEADS=16
export CLER_ENABLED=1
export CLER_GAMMA_INIT=0.01
export CLER_GAMMA_MODE=channel
export CLER_NORMALIZE_RESIDUAL=0
export CLER_RESIDUAL_NORM_EPS=1e-6
export CLER_DETACH_RESIDUAL=0
export CLER_GAMMA_LOG_INTERVAL=1
export CLER_RESIDUAL_LOG_INTERVAL=1
export RECOMPUTE=1
export MEGATRON_LINEAR_TORCH_COMPILE=1
export MEGATRON_LINEAR_TORCH_COMPILE_MODE=reduce-overhead
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=0

cd "$CLER_REPO_ROOT"
exec bash _research/launch/transformer-pp-350m-gdn-pytorch.sbatch
