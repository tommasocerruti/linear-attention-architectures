#!/bin/bash
#
# 1.3B CLER v1 PyTorch Gated DeltaNet AdamW, gamma init 1e-2.
#
# Queue with:
#   sbatch _research/launch/transformer-pp-1p3b-fwe1b-cler-v1-gamma1e-2-adamw-12h.sbatch
#
# W&B: cler/clerv1-runs, group 1p3b-fwe1b-architecture-ladder-20260506
#
#SBATCH --job-name=1p3b-fwe1b-cler-g1e2-adamw
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail

source /users/course_00252/.wandb_cler_env
if [ -f /users/course_00252/.hf_cler_env ]; then
    source /users/course_00252/.hf_cler_env
fi

export CLER_REPO_ROOT=/users/course_00252/cler
export MEGATRON_DATA_PATH=${MEGATRON_DATA_PATH:-/iopsstor/scratch/cscs/course_00252/cler/_research/results/data/fineweb_edu/fineweb_edu_1000000000_llama2/fineweb_edu_1000000000_llama2_text_document}
export LLAMA2_TOKENIZER_MODEL=${LLAMA2_TOKENIZER_MODEL:-/iopsstor/scratch/cscs/course_00252/llama2-tokenizer/tokenizer.model}
export WANDB_ENTITY=${WANDB_ENTITY:-cler}
export WANDB_PROJECT=${WANDB_PROJECT:-clerv1-runs}
export WANDB_GROUP=${WANDB_GROUP:-1p3b-fwe1b-architecture-ladder-20260506}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=${WANDB_DATASET:-fineweb_edu_1b_llama2}
export WANDB_MODEL_SIZE=1.3B
export WANDB_ARCH=cler-v1-gdn-pytorch
export WANDB_KERNEL=torch
export APERTUS_FEATURE=cler-v1-gamma1e-2-adamw
export APERTUS_TRACK=transformer-pp-1p3b-fwe1b-cler-v1-gamma1e-2
export EXP_NAME=${EXP_NAME:-1p3b-llama2-fwe1b-cler-v1-gamma1e-2-adamw}

export NUM_LAYERS=24
export HIDDEN=2048
export FFN_HIDDEN=5632
export NUM_HEADS=32
export NUM_KV_HEADS=8
export MBS=2
export GBS=128
export SEQ_LEN=4096
export TRAIN_SAMPLES=244140
export EVAL_INTERVAL=${EVAL_INTERVAL:-48828}
export EVAL_ITERS=${EVAL_ITERS:-10}
export EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-715}
export LR=2e-4
export MIN_LR=2e-5
export LINEAR_ATTENTION_FREQ=3
export LINEAR_KEY_HEAD_DIM=64
export LINEAR_VALUE_HEAD_DIM=64
export LINEAR_NUM_KEY_HEADS=8
export LINEAR_NUM_VALUE_HEADS=16
export CLER_ENABLED=1
export CLER_GAMMA_INIT=1e-2
export CLER_DETACH_RESIDUAL=0
export CLER_GAMMA_LOG_INTERVAL=${CLER_GAMMA_LOG_INTERVAL:-10}
export MEGATRON_LINEAR_TORCH_COMPILE=${MEGATRON_LINEAR_TORCH_COMPILE:-1}
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=${MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS:-0}

cd /users/course_00252/cler
exec bash _research/launch/transformer-pp-350m-gdn-pytorch.sbatch
