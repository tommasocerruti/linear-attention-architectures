#!/bin/bash
# 350M / 1B-token hybrid GDN-2 (faithful low-rank decay/gate) + Full Attention Residuals, AdamW,
# external GDN2 Triton chunk kernel. Compare to the matched GDN-2 baseline
# (transformer-pp-350m-fwe1b-gdn2-pytorch-adamw-2h58-external.sbatch).
#
# Queue with:
#   sbatch _research/launch/transformer-pp-350m-fwe1b-gdn2-attnres-adamw-2h58-external.sbatch
#
#SBATCH --job-name=gdn2-attnres-350m-adamw
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
export PACKAGE_DIR=/iopsstor/scratch/cscs/course_00252/cler/_research/packages_pure_linear_20260522
export GDN2_REPO_DIR=${GDN2_REPO_DIR:-/users/course_00252/GatedDeltaNet-2}
export MEGATRON_GDN2_USE_EXTERNAL_CHUNK=${MEGATRON_GDN2_USE_EXTERNAL_CHUNK:-1}
export MEGATRON_DATA_PATH=${MEGATRON_DATA_PATH:-/iopsstor/scratch/cscs/course_00252/cler/_research/results/data/fineweb_edu/fineweb_edu_1000000000_llama2/fineweb_edu_1000000000_llama2_text_document}
export LLAMA2_TOKENIZER_MODEL=${LLAMA2_TOKENIZER_MODEL:-/iopsstor/scratch/cscs/course_00252/llama2-tokenizer/tokenizer.model}
export WANDB_ENTITY=${WANDB_ENTITY:-cler}
export WANDB_PROJECT=${WANDB_PROJECT:-clerv1-runs}
export WANDB_GROUP=${WANDB_GROUP:-350m-fwe1b-gdn2-attnres-20260531}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=${WANDB_DATASET:-fineweb_edu_1b_llama2}
export WANDB_ARCH=${WANDB_ARCH:-gdn2-attnres}
export WANDB_KERNEL=${WANDB_KERNEL:-gdn2-triton}
export WANDB_HYBRID_MECHANISM=${WANDB_HYBRID_MECHANISM:-2gdn2-1fullcausal}
export APERTUS_FEATURE=${APERTUS_FEATURE:-gdn2-attnres-adamw}
export APERTUS_TRACK=${APERTUS_TRACK:-transformer-pp-350m-gdn2-attnres-adamw}
export EXP_NAME=${EXP_NAME:-gdn2-attnres-350m-llama2-fwe1b-adamw-external}
export GDN_ATTENTION_VARIANT=gated_delta_net2_pytorch
export ATTN_RES_ENABLED=1
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-244140}
export EVAL_INTERVAL=${EVAL_INTERVAL:-100}
export EVAL_ITERS=${EVAL_ITERS:-10}
export EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-200}
export MBS=${MBS:-2}
export GBS=${GBS:-128}
export LINEAR_ATTENTION_FREQ=${LINEAR_ATTENTION_FREQ:-3}
export MEGATRON_LINEAR_TORCH_COMPILE=${MEGATRON_LINEAR_TORCH_COMPILE:-0}
export MEGATRON_LINEAR_TORCH_COMPILE_MODE=${MEGATRON_LINEAR_TORCH_COMPILE_MODE:-reduce-overhead}
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=${MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS:-0}

cd /users/course_00252/cler
exec bash _research/launch/transformer-pp-350m-gdn-pytorch.sbatch
