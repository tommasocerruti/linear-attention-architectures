#!/bin/bash
#
# Debug smoke for the project-owned Gated DeltaNet-2 PyTorch Megatron variant.
# Runs at least 10 iterations on the 50M FineWeb-Edu/LLaMA2 prefix before any
# larger GDN2 launch should be submitted.
#
# Queue with:
#   sbatch _research/launch/debug-fwe50m-gdn2-pytorch-adamw-smoke.sbatch
#
#SBATCH --job-name=debug-gdn2-350m-a
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:58:00
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
export GDN2_REPO_DIR=${GDN2_REPO_DIR:-/users/course_00252/GatedDeltaNet-2}
export MEGATRON_GDN2_USE_EXTERNAL_CHUNK=${MEGATRON_GDN2_USE_EXTERNAL_CHUNK:-1}
export MEGATRON_DATA_PATH=${MEGATRON_DATA_PATH:-/iopsstor/scratch/cscs/course_00252/cler/_research/results/data/fineweb_edu/fineweb_edu_50m_llama2_smoke/fineweb_edu_50m_llama2_smoke_text_document}
export LLAMA2_TOKENIZER_MODEL=${LLAMA2_TOKENIZER_MODEL:-/iopsstor/scratch/cscs/course_00252/llama2-tokenizer/tokenizer.model}
export WANDB_ENTITY=${WANDB_ENTITY:-cler}
export WANDB_PROJECT=${WANDB_PROJECT:-clerv1-runs}
export WANDB_GROUP=${WANDB_GROUP:-debug-gdn2-350m-llama2-finewebedu-adamw-20260523}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=${WANDB_DATASET:-fineweb_edu_50m_llama2_smoke}
export WANDB_ARCH=${WANDB_ARCH:-gdn2-pytorch}
export WANDB_KERNEL=${WANDB_KERNEL:-gdn2-triton}
export WANDB_IS_SMOKE=1
export APERTUS_FEATURE=${APERTUS_FEATURE:-debug-gated-deltanet2-pytorch-adamw}
export APERTUS_TRACK=${APERTUS_TRACK:-debug-fwe50m-gated-deltanet2-pytorch-adamw}
export EXP_NAME=${EXP_NAME:-debug-gdn2-pytorch-350m-llama2-fwe50m-adamw-external}
export GDN_ATTENTION_VARIANT=gated_delta_net2_pytorch
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export EVAL_INTERVAL=${EVAL_INTERVAL:-10}
export EVAL_ITERS=${EVAL_ITERS:-0}
export EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-50}
export MBS=${MBS:-2}
export GBS=${GBS:-128}
export LINEAR_ATTENTION_FREQ=${LINEAR_ATTENTION_FREQ:-3}
export MEGATRON_LINEAR_TORCH_COMPILE=${MEGATRON_LINEAR_TORCH_COMPILE:-0}
export MEGATRON_LINEAR_TORCH_COMPILE_MODE=${MEGATRON_LINEAR_TORCH_COMPILE_MODE:-reduce-overhead}
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=${MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS:-0}

cd /users/course_00252/cler
exec bash _research/launch/transformer-pp-350m-gdn-pytorch.sbatch
