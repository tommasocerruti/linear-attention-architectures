#!/bin/bash
#
# 350M CLER v1 Muon with cross-layer residual carry-through across non-CLER layers,
# torch.compile without Inductor CUDAGraphs, FineWeb-Edu 1B / LLaMA2 tokenizer.
#
# Queue with:
#   sbatch _research/launch/transformer-pp-350m-fwe1b-cler-v1-carry-muon-10h-compile.sbatch
#
# W&B: cler/clerv1-runs, group 350m-llama2-finewebedu-1b-pytorch-muon-10h-20260503
#
#SBATCH --job-name=350m-fwe1b-cler-carry-muon-c10h
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

cd /users/course_00252/cler
export CLER_REPO_ROOT=/users/course_00252/cler
export WANDB_GROUP=${WANDB_GROUP:-350m-llama2-finewebedu-1b-pytorch-muon-10h-20260503}
export EXP_NAME=${EXP_NAME:-350m-llama2-fwe1b-cler-v1-carry-muon-compile10h}
export EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-585}
export EVAL_INTERVAL=${EVAL_INTERVAL:-100}
export EVAL_ITERS=${EVAL_ITERS:-10}
export MEGATRON_LINEAR_TORCH_COMPILE=${MEGATRON_LINEAR_TORCH_COMPILE:-1}
export MEGATRON_LINEAR_TORCH_COMPILE_MODE=${MEGATRON_LINEAR_TORCH_COMPILE_MODE:-reduce-overhead}
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=${MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS:-0}
exec _research/launch/run_350m_finewebedu_llama2_compare.sh cler_muon
