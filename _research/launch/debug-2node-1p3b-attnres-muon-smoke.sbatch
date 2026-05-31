#!/bin/bash
# OUR-account 2-node / 8-GPU multi-node bring-up smoke for the 1.3B AttnRes path (10 iters).
# Validates that the 1.3B hybrid GDN-Triton + AttnRes code runs multi-node (so the teammate's
# 16-node runs only need his account/paths, no debugging). Uses our 50M smoke data + our W&B.
#
#SBATCH --job-name=debug-2node-1p3b-attnres
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=2
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
export MEGATRON_DATA_PATH=/iopsstor/scratch/cscs/course_00252/cler/_research/results/data/fineweb_edu/fineweb_edu_50m_llama2_smoke/fineweb_edu_50m_llama2_smoke_text_document
export LLAMA2_TOKENIZER_MODEL=/iopsstor/scratch/cscs/course_00252/llama2-tokenizer/tokenizer.model

export WANDB_ENTITY=cler
export WANDB_PROJECT=clerv1-runs
export WANDB_GROUP=debug-2node-1p3b-20260531
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=fineweb_edu_50m_llama2_smoke
export WANDB_ARCH=1p3b-hybrid-gdn-attnres
export WANDB_KERNEL=triton
export WANDB_IS_SMOKE=1
export APERTUS_FEATURE=debug-2node-1p3b-attnres
export APERTUS_TRACK=debug-2node-1p3b-attnres
export EXP_NAME=debug-2node-1p3b-attnres-muon-smoke

# 1.3B config (matches the teammate's 1.3B/40B launchers)
export NUM_LAYERS=24
export HIDDEN=2048
export FFN_HIDDEN=5632
export NUM_HEADS=16
export NUM_KV_HEADS=8
export LINEAR_KEY_HEAD_DIM=64
export LINEAR_VALUE_HEAD_DIM=64
export LINEAR_NUM_KEY_HEADS=8
export LINEAR_NUM_VALUE_HEADS=16
export LINEAR_ATTENTION_FREQ=3

export LINEAR_VARIANT=gdn_triton
export ATTN_RES_ENABLED=1

export TRAIN_ITERS=10
export EVAL_INTERVAL=10
export EVAL_ITERS=0
export EXIT_DURATION_IN_MINS=25
export MBS=2
export GBS=64
export ENABLE_CHECKPOINTING=0

# multi-node safety (avoid NCCL init hang / Inductor fuser issues at scale)
export MEGATRON_DISABLE_JIT_FUSER=1
export MEGATRON_INIT_PROCESS_GROUP_DEVICE_ID=1
export MEGATRON_LINEAR_TORCH_COMPILE=1
export MEGATRON_LINEAR_TORCH_COMPILE_MODE=reduce-overhead
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=0

cd /users/course_00252/cler
exec bash _research/launch/transformer-pp-350m-linear-muon.sbatch
