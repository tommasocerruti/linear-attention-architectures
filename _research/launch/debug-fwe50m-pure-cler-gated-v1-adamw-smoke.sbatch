#!/bin/bash
#
# Debug smoke for pure scalar CLER-Gated on FineWeb-Edu 50M / LLaMA2.
# Pure means every one of the 20 mixer layers is GDN: LINEAR_ATTENTION_FREQ=21.
#
# Queue with:
#   sbatch _research/launch/debug-fwe50m-pure-cler-gated-v1-adamw-smoke.sbatch
#
#SBATCH --job-name=dbg-pure-cler-a
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:45:00
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
export PACKAGE_DIR=/iopsstor/scratch/cscs/course_00252/cler/_research/packages_pure_linear_20260522
export MEGATRON_DATA_PATH=/iopsstor/scratch/cscs/course_00252/cler/_research/results/data/fineweb_edu/fineweb_edu_50m_llama2_smoke/fineweb_edu_50m_llama2_smoke_text_document
export LLAMA2_TOKENIZER_MODEL=/iopsstor/scratch/cscs/course_00252/llama2-tokenizer/tokenizer.model
export WANDB_ENTITY=cler
export WANDB_PROJECT=clerv1-runs
export WANDB_GROUP=debug-fwe50m-pure-linear-adamw-20260522
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=fineweb_edu_50m_llama2_smoke
export WANDB_IS_SMOKE=1
export WANDB_ARCH=pure-cler-v1-gdn-pytorch
export WANDB_KERNEL=torch
export APERTUS_FEATURE=debug-pure-cler-v1-gdn-pytorch-adamw
export APERTUS_TRACK=debug-fwe50m-pure-cler-v1-gdn-pytorch-adamw
export EXP_NAME=debug-pure-cler-gated-v1-scalar-350m-llama2-fwe50m-adamw
export TRAIN_ITERS=8
export LR_WARMUP_ITERS=1
export EVAL_INTERVAL=8
export EVAL_ITERS=0
export EXIT_DURATION_IN_MINS=35
export MBS=2
export GBS=128
export LINEAR_ATTENTION_FREQ=21
export CLER_ENABLED=1
export CLER_GAMMA_INIT=0.0
export CLER_GAMMA_MODE=scalar
export CLER_NORMALIZE_RESIDUAL=0
export CLER_GAMMA_LOG_INTERVAL=1
export CLER_RESIDUAL_LOG_INTERVAL=1
export MEGATRON_LINEAR_TORCH_COMPILE=1
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=0

cd /users/course_00252/cler
exec bash _research/launch/transformer-pp-350m-gdn-pytorch.sbatch
