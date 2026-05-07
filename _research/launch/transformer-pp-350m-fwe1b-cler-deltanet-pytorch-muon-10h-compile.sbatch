#!/bin/bash
#
# 350M CLER-on-DeltaNet PyTorch Muon run on FineWeb-Edu 1B / LLaMA2.
#
# Submit with:
#   sbatch _research/launch/transformer-pp-350m-fwe1b-cler-deltanet-pytorch-muon-10h-compile.sbatch
#
# This wrapper delegates to the shared Linear-RNN Muon launcher and only pins
# the dataset, W&B labels, CLER variant, and run length.
#
#SBATCH --job-name=350m-fwe1b-cler-dn
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=10:00:00
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
export WANDB_GROUP=350m-llama2-finewebedu-1b-pytorch-muon-10h-20260507
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=fineweb_edu_1b_llama2
export WANDB_ARCH=cler-v1-deltanet-pytorch
export WANDB_KERNEL=torch
export APERTUS_FEATURE=cler-v1-deltanet-pytorch-muon-compile10h
export APERTUS_TRACK=transformer-pp-350m-cler-deltanet-pytorch-muon-10h
export LINEAR_VARIANT=cler_deltanet_pytorch
export EXP_NAME=350m-llama2-fwe1b-cler-deltanet-pytorch-muon-compile10h
export TRAIN_SAMPLES=244140
export EVAL_INTERVAL=100
export EVAL_ITERS=10
export EXIT_DURATION_IN_MINS=585
export MBS=2
export GBS=128
export CLER_GAMMA_INIT=0.0
export CLER_GAMMA_LOG_INTERVAL=10
export MEGATRON_LINEAR_TORCH_COMPILE=1
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=0

cd /users/course_00252/cler
exec bash _research/launch/transformer-pp-350m-linear-muon.sbatch
