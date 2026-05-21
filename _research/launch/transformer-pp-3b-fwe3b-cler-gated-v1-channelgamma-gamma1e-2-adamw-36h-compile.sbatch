#!/bin/bash
#
# 3B-ish CLER-Gated PyTorch AdamW scale probe on FineWeb-Edu / LLaMA2.
# Uses the repo's established 2.7B dense shape: 32L / 2560H / 7680FFN.
# CLER: per-channel receiver gamma, raw routed residual, gamma init 1e-2.
# Token budget: TRAIN_SAMPLES=732422 ~= 3.0B tokens at seq_len 4096.
#
# Queue with:
#   sbatch _research/launch/transformer-pp-3b-fwe3b-cler-gated-v1-channelgamma-gamma1e-2-adamw-36h-compile.sbatch
#
# Run smoke first:
#   sbatch _research/launch/debug-fwe50m-3b-cler-gated-channelgamma-adamw-smoke.sbatch
#
# If running from another account/checkout, override:
#   CLER_REPO_ROOT=/path/to/cler
#   MEGATRON_DATA_PATH=/path/to/fineweb_edu_1000000000_llama2_text_document
#   LLAMA2_TOKENIZER_MODEL=/path/to/tokenizer.model
#
#SBATCH --job-name=3b-fwe3b-cler-cg-a
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=36:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=16
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

export MEGATRON_DATA_PATH=${MEGATRON_DATA_PATH:-/iopsstor/scratch/cscs/course_00252/cler/_research/results/data/fineweb_edu/fineweb_edu_1000000000_llama2/fineweb_edu_1000000000_llama2_text_document}
export LLAMA2_TOKENIZER_MODEL=${LLAMA2_TOKENIZER_MODEL:-/iopsstor/scratch/cscs/course_00252/llama2-tokenizer/tokenizer.model}
export WANDB_ENTITY=${WANDB_ENTITY:-cler}
export WANDB_PROJECT=${WANDB_PROJECT:-clerv1-runs}
export WANDB_GROUP=${WANDB_GROUP:-3b-fwe3b-gdn-vs-cler-channelgamma-20260518}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=${WANDB_DATASET:-fineweb_edu_1b_llama2_repeated_to_3b}
export WANDB_MODEL_SIZE=3B
export WANDB_ARCH=cler-v1-gdn-pytorch-channelgamma
export WANDB_KERNEL=torch
export APERTUS_FEATURE=3b-cler-gated-channelgamma-gamma1e-2-adamw-compile36h
export APERTUS_TRACK=transformer-pp-3b-fwe3b-cler-gated-channelgamma-adamw
export EXP_NAME=${EXP_NAME:-3b-llama2-fwe3b-cler-gated-channelgamma-adamw-gamma1e-2-compile36h}

export NUM_LAYERS=32
export HIDDEN=2560
export FFN_HIDDEN=7680
export NUM_HEADS=32
export NUM_KV_HEADS=8
export MBS=1
export GBS=128
export SEQ_LEN=4096
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-732422}
export EVAL_INTERVAL=${EVAL_INTERVAL:-500}
export EVAL_ITERS=${EVAL_ITERS:-10}
export EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-2145}
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
export CLER_GAMMA_LOG_INTERVAL=100
export CLER_RESIDUAL_LOG_INTERVAL=100
export RECOMPUTE=1
export MEGATRON_LINEAR_TORCH_COMPILE=1
export MEGATRON_LINEAR_TORCH_COMPILE_MODE=reduce-overhead
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=0

cd "$CLER_REPO_ROOT"
exec bash _research/launch/transformer-pp-350m-gdn-pytorch.sbatch
