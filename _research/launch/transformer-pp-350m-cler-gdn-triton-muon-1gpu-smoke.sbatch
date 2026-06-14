#!/bin/bash
#
# Triton/FLA CLER Gated DeltaNet ~350M on 1 GPU, 500 iterations, Muon optimizer.
# Use with the PyTorch CLER smoke launcher to compare loss curves and throughput.
#
#SBATCH --job-name=cler-gdn-triton-1gpu-smoke
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail

REPO_ROOT=${CLER_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/iopsstor/scratch/cscs/$USER/cler}}
if [ ! -f "$REPO_ROOT/pretrain_gpt.py" ]; then
    REPO_ROOT=/iopsstor/scratch/cscs/$USER/cler
fi
cd "$REPO_ROOT"
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

export LINEAR_VARIANT=cler_gdn_triton
export WANDB_KERNEL=triton
export WANDB_ARCH=${WANDB_ARCH:-cler-v1-gdn-triton}
export WANDB_IS_SMOKE=1
export APERTUS_FEATURE=${APERTUS_FEATURE:-cler-v1-gdn-triton-1gpu-smoke}
export APERTUS_TRACK=${APERTUS_TRACK:-cler-gdn-triton-1gpu-smoke}
export EXP_NAME=${EXP_NAME:-cler-gdn-triton-1gpu-smoke-$(git rev-parse --short HEAD)}

export TP=1
export PP=1
export CP=1
export MBS=${MBS:-16}
export GBS=${GBS:-16}
export TRAIN_ITERS=${TRAIN_ITERS:-500}
export EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-115}
export EVAL_INTERVAL=${EVAL_INTERVAL:-$TRAIN_ITERS}
export EVAL_ITERS=${EVAL_ITERS:-0}
export LINEAR_ATTENTION_FREQ=${LINEAR_ATTENTION_FREQ:-"'([1]*20)'"}

exec bash _research/launch/transformer-pp-350m-linear-muon.sbatch
