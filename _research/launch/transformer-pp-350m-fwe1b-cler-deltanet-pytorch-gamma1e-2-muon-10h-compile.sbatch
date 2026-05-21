#!/bin/bash
#
# 350M CLER-DeltaNet PyTorch Muon gamma-init probe on FineWeb-Edu 1B / LLaMA2.
# Gamma init: 0.01.
#
# Submit with:
#   sbatch _research/launch/transformer-pp-350m-fwe1b-cler-deltanet-pytorch-gamma1e-2-muon-10h-compile.sbatch
#
#SBATCH --job-name=cler-dn-g1e2-350m
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
export MEGATRON_DATA_PATH=/iopsstor/scratch/cscs/course_00252/cler/_research/results/data/fineweb_edu/fineweb_edu_1000000000_llama2/fineweb_edu_1000000000_llama2_text_document
export LLAMA2_TOKENIZER_MODEL=/iopsstor/scratch/cscs/course_00252/llama2-tokenizer/tokenizer.model
export WANDB_ENTITY=cler
export WANDB_PROJECT=clerv1-runs
export WANDB_GROUP=350m-llama2-finewebedu-1b-cler-gamma-init-20260511
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=fineweb_edu_1b_llama2
export WANDB_ARCH=cler-v1-deltanet-pytorch
export WANDB_KERNEL=torch
export APERTUS_FEATURE=cler-deltanet-pytorch-gamma1e-2-muon-compile10h
export APERTUS_TRACK=transformer-pp-350m-cler-deltanet-pytorch-gamma-init-muon-10h
export LINEAR_VARIANT=cler_deltanet_pytorch
export EXP_NAME=cler-deltanet-pytorch-350m-llama2-fwe1b-muon-gamma1e-2-compile10h
export TRAIN_SAMPLES=244140
export EVAL_INTERVAL=100
export EVAL_ITERS=10
export EXIT_DURATION_IN_MINS=585
export MBS=2
export GBS=128
export CLER_GAMMA_INIT=0.01
export CLER_GAMMA_LOG_INTERVAL=10
export CLER_RESIDUAL_LOG_INTERVAL=10
export MEGATRON_LINEAR_TORCH_COMPILE=1
export MEGATRON_LINEAR_TORCH_COMPILE_MODE=reduce-overhead
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=0

cd /users/course_00252/cler
exec bash _research/launch/transformer-pp-350m-linear-muon.sbatch

