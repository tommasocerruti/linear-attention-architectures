#!/bin/bash
#
# 350M CLER-on-GDN PyTorch AdamW, torch.compile, FineWeb-Edu 1B / LLaMA2.
# Uses scalar gamma, raw routed residual, and corrected carry-through.
#
# Queue with:
#   sbatch _research/launch/transformer-pp-350m-fwe1b-cler-gated-v1-adamw-9h58-compile.sbatch
#
#SBATCH --job-name=cler-g-350m-a
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=09:58:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
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
export WANDB_GROUP=${WANDB_GROUP:-350m-llama2-finewebedu-1b-adamw-cler-effects-20260514}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=${WANDB_DATASET:-fineweb_edu_1b_llama2}
export WANDB_ARCH=${WANDB_ARCH:-cler-v1-gdn-pytorch}
export WANDB_KERNEL=${WANDB_KERNEL:-torch}
export APERTUS_FEATURE=${APERTUS_FEATURE:-cler-v1-gdn-pytorch-adamw}
export APERTUS_TRACK=${APERTUS_TRACK:-transformer-pp-350m-cler-v1-adamw}
export EXP_NAME=${EXP_NAME:-cler-gated-v1-carry-350m-llama2-fwe1b-adamw-compile10h}
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-244140}
export EVAL_INTERVAL=${EVAL_INTERVAL:-100}
export EVAL_ITERS=${EVAL_ITERS:-10}
export EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-585}
export MBS=${MBS:-2}
export GBS=${GBS:-128}
export CLER_ENABLED=1
export CLER_GAMMA_INIT=${CLER_GAMMA_INIT:-0.0}
export CLER_GAMMA_MODE=${CLER_GAMMA_MODE:-scalar}
export CLER_NORMALIZE_RESIDUAL=0
export CLER_GAMMA_LOG_INTERVAL=${CLER_GAMMA_LOG_INTERVAL:-20}
export CLER_RESIDUAL_LOG_INTERVAL=${CLER_RESIDUAL_LOG_INTERVAL:-20}
export MEGATRON_LINEAR_TORCH_COMPILE=${MEGATRON_LINEAR_TORCH_COMPILE:-1}
export MEGATRON_LINEAR_TORCH_COMPILE_MODE=${MEGATRON_LINEAR_TORCH_COMPILE_MODE:-reduce-overhead}
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=${MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS:-0}

cd /users/course_00252/cler
exec bash _research/launch/transformer-pp-350m-gdn-pytorch.sbatch
