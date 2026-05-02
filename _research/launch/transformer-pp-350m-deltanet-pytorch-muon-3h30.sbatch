#!/bin/bash
#
# 350M pure PyTorch DeltaNet Muon on FineWeb-Edu 1B / LLaMA2 tokenizer.
#
# Queue with:
#   sbatch _research/launch/transformer-pp-350m-deltanet-pytorch-muon-3h30.sbatch
#
# W&B: cler/clerv1-runs, group 350m-llama2-finewebedu-1b-deltanet-pytorch-20260502
#
#SBATCH --job-name=350m-fwe1b-dn-pytorch-muon
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=03:30:00
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
export WANDB_GROUP=${WANDB_GROUP:-350m-llama2-finewebedu-1b-deltanet-pytorch-20260502}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=${WANDB_DATASET:-fineweb_edu_1b_llama2}
export WANDB_ARCH=${WANDB_ARCH:-dn-pytorch}
export WANDB_KERNEL=${WANDB_KERNEL:-torch}
export LINEAR_VARIANT=deltanet_pytorch
export EXP_NAME=${EXP_NAME:-350m-llama2-fwe1b-deltanet-pytorch-muon}
export TRAIN_ITERS=${TRAIN_ITERS:-200}
export EVAL_INTERVAL=${EVAL_INTERVAL:-100}
export EVAL_ITERS=${EVAL_ITERS:-2}
export EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-200}
export MBS=${MBS:-2}
export GBS=${GBS:-128}
export MEGATRON_LINEAR_TORCH_COMPILE=${MEGATRON_LINEAR_TORCH_COMPILE:-1}
export MEGATRON_LINEAR_TORCH_COMPILE_MODE=${MEGATRON_LINEAR_TORCH_COMPILE_MODE:-reduce-overhead}

cd /users/course_00252/cler
exec bash _research/launch/transformer-pp-350m-linear-muon.sbatch
