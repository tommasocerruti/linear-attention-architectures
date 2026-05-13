#!/bin/bash
#
# Debug smoke for CLER-Gated per-head-gamma ablation on FineWeb-Edu 50M / LLaMA2.
#
# Submit with:
#   sbatch _research/launch/debug-fwe50m-cler-gated-headgamma-gamma1e-2-muon-smoke.sbatch
#
#SBATCH --job-name=dbg-cler-g-hg
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
export WANDB_GROUP=debug-fwe50m-cler-gamma-norm-ablation-20260513
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=fineweb_edu_50m_llama2_smoke
export WANDB_IS_SMOKE=1
export WANDB_ARCH=cler-v1-gdn-pytorch-headgamma
export WANDB_KERNEL=torch
export APERTUS_FEATURE=debug-cler-gated-v1-headgamma-gamma1e-2-muon
export APERTUS_TRACK=debug-fwe50m-cler-gated-v1-gamma-norm-ablation
export LINEAR_VARIANT=cler
export EXP_NAME=debug-cler-gated-v1-headgamma-350m-llama2-fwe50m-muon-gamma1e-2-compile
export TRAIN_ITERS=8
export LR_WARMUP_ITERS=1
export EVAL_INTERVAL=8
export EVAL_ITERS=0
export EXIT_DURATION_IN_MINS=35
export MBS=2
export GBS=128
export CLER_GAMMA_INIT=0.01
export CLER_GAMMA_MODE=head
export CLER_NORMALIZE_RESIDUAL=0
export CLER_GAMMA_LOG_INTERVAL=1
export CLER_RESIDUAL_LOG_INTERVAL=1
export MEGATRON_LINEAR_TORCH_COMPILE=1
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=0

cd /users/course_00252/cler
exec bash _research/launch/transformer-pp-350m-linear-muon.sbatch
