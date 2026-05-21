#!/bin/bash
#
# Full 350M DeltaNet PyTorch Muon baseline on FineWeb-Edu 1B / LLaMA2.
#
# Submit with:
#   sbatch _research/launch/transformer-pp-350m-fwe1b-deltanet-pytorch-muon-10h-compile.sbatch
#
# This is the matched baseline for CLER-DeltaNet PyTorch. It uses the shared
# Linear-RNN Muon launcher, torch.compile, 1B-token train target, and a 9:58
# walltime to avoid the exact 10h queue bucket.
#
#SBATCH --job-name=dn-350m-fwe1b
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
export WANDB_GROUP=${WANDB_GROUP:-350m-llama2-finewebedu-1b-pytorch-muon-10h-20260508}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=${WANDB_DATASET:-fineweb_edu_1b_llama2}
export WANDB_ARCH=${WANDB_ARCH:-deltanet-pytorch}
export WANDB_KERNEL=${WANDB_KERNEL:-torch}
export APERTUS_FEATURE=${APERTUS_FEATURE:-deltanet-pytorch-muon-compile10h}
export APERTUS_TRACK=${APERTUS_TRACK:-transformer-pp-350m-deltanet-pytorch-muon-10h}
export LINEAR_VARIANT=deltanet_pytorch
export EXP_NAME=${EXP_NAME:-350m-llama2-fwe1b-deltanet-pytorch-muon-compile10h}
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-244140}
export EVAL_INTERVAL=${EVAL_INTERVAL:-100}
export EVAL_ITERS=${EVAL_ITERS:-10}
export EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-585}
export MBS=${MBS:-2}
export GBS=${GBS:-128}
export MEGATRON_LINEAR_TORCH_COMPILE=${MEGATRON_LINEAR_TORCH_COMPILE:-1}
export MEGATRON_LINEAR_TORCH_COMPILE_MODE=${MEGATRON_LINEAR_TORCH_COMPILE_MODE:-reduce-overhead}
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=${MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS:-0}

cd /users/course_00252/cler
exec bash _research/launch/transformer-pp-350m-linear-muon.sbatch

