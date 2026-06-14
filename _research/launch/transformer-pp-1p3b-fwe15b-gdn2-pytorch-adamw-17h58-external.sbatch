#!/bin/bash
#
# 1.3B-backbone hybrid Gated DeltaNet-2 AdamW scale probe.
# Token budget: 15B tokens at seq_len 4096.
# Resources: 4 nodes / 16 GH200 GPUs, TP=1, PP=1, DP=16.
#
# Queue with:
#   sbatch _research/launch/transformer-pp-1p3b-fwe15b-gdn2-pytorch-adamw-17h58-external.sbatch
#
# Smoke first:
#   sbatch _research/launch/debug-fwe50m-1p3b-gdn2-pytorch-adamw-smoke.sbatch
#
# Dataset behavior:
# - If a real 15B FineWeb-Edu/LLaMA2 Megatron prefix exists at the standard
#   scratch path, this wrapper uses it.
# - On course_00252 at creation time only the 1B prefix exists, so the default
#   falls back to training 15B tokens over the 1B prefix repeated by the
#   dataloader. Override MEGATRON_DATA_PATH if a teammate has the real 15B
#   prefix in another scratch location.
#
# If running from another checkout/account, override:
#   CLER_REPO_ROOT=/path/to/cler
#   GDN2_REPO_DIR=/path/to/GatedDeltaNet-2
#   MEGATRON_DATA_PATH=/path/to/fineweb_edu_15b_llama2_text_document
#   LLAMA2_TOKENIZER_MODEL=/path/to/tokenizer.model
#
#SBATCH --job-name=1p3b-fwe15b-gdn2-a
#SBATCH --account=lsaie-ss26
#SBATCH --partition=low
#SBATCH --time=17:58:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail

REPO_ROOT_DEFAULT=${SLURM_SUBMIT_DIR:-/users/course_00252/cler}
export CLER_REPO_ROOT=${CLER_REPO_ROOT:-$REPO_ROOT_DEFAULT}
if [ ! -f "$CLER_REPO_ROOT/pretrain_gpt.py" ] && [ -f /users/course_00252/cler/pretrain_gpt.py ]; then
    export CLER_REPO_ROOT=/users/course_00252/cler
fi

WANDB_ENV_FILE=${WANDB_ENV_FILE:-$HOME/.wandb_cler_env}
HF_ENV_FILE=${HF_ENV_FILE:-$HOME/.hf_cler_env}
if [ -f "$WANDB_ENV_FILE" ]; then
    source "$WANDB_ENV_FILE"
fi
if [ -f "$HF_ENV_FILE" ]; then
    source "$HF_ENV_FILE"
fi

GDN2_REPO_CANDIDATE=${GDN2_REPO_DIR:-$HOME/GatedDeltaNet-2}
if [ ! -f "$GDN2_REPO_CANDIDATE/lit_gpt/gdn2_ops/chunk_gdn2.py" ] && [ -f /users/course_00252/GatedDeltaNet-2/lit_gpt/gdn2_ops/chunk_gdn2.py ]; then
    GDN2_REPO_CANDIDATE=/users/course_00252/GatedDeltaNet-2
fi
export GDN2_REPO_DIR=$GDN2_REPO_CANDIDATE
export MEGATRON_GDN2_USE_EXTERNAL_CHUNK=${MEGATRON_GDN2_USE_EXTERNAL_CHUNK:-1}

export PACKAGE_DIR=${PACKAGE_DIR:-/iopsstor/scratch/cscs/course_00252/cler/_research/packages_pure_linear_20260522}

FWE15_PREFIX=/iopsstor/scratch/cscs/course_00252/cler/_research/results/data/fineweb_edu/fineweb_edu_15b_llama2/fineweb_edu_15b_llama2_text_document
FWE1_PREFIX=/iopsstor/scratch/cscs/course_00252/cler/_research/results/data/fineweb_edu/fineweb_edu_1000000000_llama2/fineweb_edu_1000000000_llama2_text_document
if [ -z "${MEGATRON_DATA_PATH:-}" ]; then
    if [ -f "${FWE15_PREFIX}.idx" ]; then
        export MEGATRON_DATA_PATH=$FWE15_PREFIX
        export WANDB_DATASET=${WANDB_DATASET:-fineweb_edu_15b_llama2}
    else
        export MEGATRON_DATA_PATH=$FWE1_PREFIX
        export WANDB_DATASET=${WANDB_DATASET:-fineweb_edu_1b_llama2_repeated_to_15b}
        echo "WARNING: 15B FineWeb-Edu/LLaMA2 prefix not found; using 1B prefix repeated for a 15B-token budget." >&2
    fi
fi

export LLAMA2_TOKENIZER_MODEL=${LLAMA2_TOKENIZER_MODEL:-/iopsstor/scratch/cscs/course_00252/llama2-tokenizer/tokenizer.model}
export WANDB_ENTITY=${WANDB_ENTITY:-cler}
export WANDB_PROJECT=${WANDB_PROJECT:-clerv1-runs}
export WANDB_GROUP=${WANDB_GROUP:-1p3b-fwe15b-gdn2-adamw-20260523}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_MODEL_SIZE=1.3B-backbone
export WANDB_ARCH=gdn2-pytorch
export WANDB_KERNEL=gdn2-triton
export WANDB_HYBRID_MECHANISM=2gdn2-1fullcausal
export APERTUS_FEATURE=1p3b-gated-deltanet2-pytorch-adamw-external18h
export APERTUS_TRACK=transformer-pp-1p3b-fwe15b-gdn2-pytorch-adamw
export EXP_NAME=${EXP_NAME:-1p3b-llama2-fwe15b-gdn2-pytorch-adamw-external18h}

export GDN_ATTENTION_VARIANT=gated_delta_net2_pytorch
export NUM_LAYERS=24
export HIDDEN=2048
export FFN_HIDDEN=5632
export NUM_HEADS=32
export NUM_KV_HEADS=8
export MBS=2
export GBS=128
export SEQ_LEN=4096
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-3662109}
export EVAL_INTERVAL=${EVAL_INTERVAL:-2000}
export EVAL_ITERS=${EVAL_ITERS:-10}
export EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-1075}
export LR=2e-4
export MIN_LR=2e-5
export TP=${TP:-1}
export PP=${PP:-1}
export LINEAR_ATTENTION_FREQ=3
export LINEAR_KEY_HEAD_DIM=128
export LINEAR_VALUE_HEAD_DIM=128
export LINEAR_NUM_KEY_HEADS=16
export LINEAR_NUM_VALUE_HEADS=16
export CLER_ENABLED=0
export RECOMPUTE=1
export RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-24}
export MEGATRON_LINEAR_TORCH_COMPILE=0
export MEGATRON_LINEAR_TORCH_COMPILE_MODE=reduce-overhead
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=0
export MEGATRON_DISABLE_JIT_FUSER=1
export MEGATRON_INIT_PROCESS_GROUP_DEVICE_ID=1

cd "$CLER_REPO_ROOT"
exec bash _research/launch/transformer-pp-350m-gdn-pytorch.sbatch
