#!/bin/bash
#
# Debug smoke for additive Linear Transformer PyTorch + Muon on FineWeb-Edu 50M / LLaMA2.
#
# Submit with:
#   sbatch _research/launch/debug-fwe50m-linear-transformer-pytorch-muon-smoke.sbatch
#
# This exercises the same shared Muon wrapper used by the 350M full run.
#
#SBATCH --job-name=debug-fwe50m-lt-muon
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
export MEGATRON_DATA_PATH=/iopsstor/scratch/cscs/course_00252/cler/_research/results/data/fineweb_edu/fineweb_edu_50m_llama2_smoke/fineweb_edu_50m_llama2_smoke_text_document
export LLAMA2_TOKENIZER_MODEL=/iopsstor/scratch/cscs/course_00252/llama2-tokenizer/tokenizer.model
export WANDB_ENTITY=cler
export WANDB_PROJECT=clerv1-runs
export WANDB_GROUP=debug-fwe50m-linear-transformer-muon-20260507
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=fineweb_edu_50m_llama2_smoke
export WANDB_IS_SMOKE=1
export WANDB_ARCH=linear-transformer-pytorch
export WANDB_KERNEL=torch
export APERTUS_FEATURE=debug-linear-transformer-pytorch-muon
export APERTUS_TRACK=debug-fwe50m-linear-transformer-pytorch-muon
export LINEAR_VARIANT=linear_transformer_pytorch
export EXP_NAME=debug-fwe50m-linear-transformer-pytorch-muon
export TRAIN_ITERS=8
export LR_WARMUP_ITERS=1
export EVAL_INTERVAL=8
export EVAL_ITERS=0
export EXIT_DURATION_IN_MINS=35
export MBS=2
export GBS=128
export MEGATRON_LINEAR_TORCH_COMPILE=1
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=0

cd /users/course_00252/cler
exec bash _research/launch/transformer-pp-350m-linear-muon.sbatch
