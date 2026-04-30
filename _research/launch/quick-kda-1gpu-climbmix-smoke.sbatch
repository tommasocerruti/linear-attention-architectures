#!/bin/bash
#
# Short KDA AdamW runtime smoke on ClimbMix.
#
#SBATCH --job-name=quick-kda-1gpu-climbmix-smoke
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000
#SBATCH --no-requeue

set -euo pipefail

SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* Command=\([^ ]*\).*/\1/p')
if [ -n "$SCRIPT_PATH" ]; then
    REPO_ROOT=$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)
else
    REPO_ROOT=${CLER_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
fi
SCRIPT_DIR=$REPO_ROOT/_research/launch
cd "$REPO_ROOT"

export TRAIN_ITERS=${TRAIN_ITERS:-50}
export EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-20}
export EVAL_INTERVAL=${EVAL_INTERVAL:-25}
export EVAL_ITERS=${EVAL_ITERS:-1}
export WANDB_IS_SMOKE=1
GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
export EXP_NAME=${EXP_NAME:-quick-kda-1gpu-climbmix-smoke-$GIT_SHA}
export APERTUS_TRACK=${APERTUS_TRACK:-quick-kda-1gpu-smoke}

exec bash "$SCRIPT_DIR/quick-kda-1gpu-climbmix.sbatch"
