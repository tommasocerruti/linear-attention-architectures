#!/bin/bash
#
# Debug smoke for 1.3B-backbone Gated DeltaNet-2 on 4 debug nodes.
# This validates the same 4-node / 16-GPU distributed shape intended by the
# 15B-token launcher, but stops after 10 iterations on the 50M smoke prefix.
#
# Queue with:
#   sbatch _research/launch/debug-fwe50m-1p3b-gdn2-pytorch-adamw-smoke.sbatch
#
# If running from another checkout/account, override:
#   CLER_REPO_ROOT=/path/to/cler
#   GDN2_REPO_DIR=/path/to/GatedDeltaNet-2
#   MEGATRON_DATA_PATH=/path/to/fineweb_edu_50m_llama2_smoke_text_document
#   LLAMA2_TOKENIZER_MODEL=/path/to/tokenizer.model
#
#SBATCH --job-name=dbg-1p3b-gdn2-a
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:14:00
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
export MEGATRON_DATA_PATH=${MEGATRON_DATA_PATH:-/iopsstor/scratch/cscs/course_00252/cler/_research/results/data/fineweb_edu/fineweb_edu_50m_llama2_smoke/fineweb_edu_50m_llama2_smoke_text_document}
export LLAMA2_TOKENIZER_MODEL=${LLAMA2_TOKENIZER_MODEL:-/iopsstor/scratch/cscs/course_00252/llama2-tokenizer/tokenizer.model}
export WANDB_ENTITY=${WANDB_ENTITY:-cler}
export WANDB_PROJECT=${WANDB_PROJECT:-clerv1-runs}
export WANDB_GROUP=${WANDB_GROUP:-debug-fwe50m-1p3b-gdn2-adamw-20260523}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=fineweb_edu_50m_llama2_smoke
export WANDB_IS_SMOKE=1
export WANDB_MODEL_SIZE=1.3B-backbone
export WANDB_ARCH=gdn2-pytorch
export WANDB_KERNEL=gdn2-triton
export WANDB_HYBRID_MECHANISM=2gdn2-1fullcausal
export APERTUS_FEATURE=debug-1p3b-gated-deltanet2-pytorch-adamw
export APERTUS_TRACK=debug-fwe50m-1p3b-gdn2-pytorch-adamw
export EXP_NAME=debug-1p3b-llama2-fwe50m-gdn2-pytorch-adamw-external

export GDN_ATTENTION_VARIANT=gated_delta_net2_pytorch
export NUM_LAYERS=24
export HIDDEN=2048
export FFN_HIDDEN=5632
export NUM_HEADS=32
export NUM_KV_HEADS=8
export MBS=2
export GBS=128
export SEQ_LEN=4096
export TRAIN_ITERS=10
export LR_WARMUP_ITERS=2
export EVAL_INTERVAL=10
export EVAL_ITERS=0
export EXIT_DURATION_IN_MINS=12
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
