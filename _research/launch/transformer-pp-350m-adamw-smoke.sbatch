#!/bin/bash
#
# Short AdamW runtime smoke for the FineWeb-Edu/LLaMA-2 path.
# Reuses the full 350M AdamW launcher with a small stopping condition.
#
#SBATCH --job-name=transformer-pp-350m-adamw-smoke
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

export TRAIN_ITERS=${TRAIN_ITERS:-50}
export EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-25}
export EVAL_INTERVAL=${EVAL_INTERVAL:-25}
export EVAL_ITERS=${EVAL_ITERS:-1}
export EXP_NAME=${EXP_NAME:-transformer-pp-350m-adamw-smoke-$(git -C "$REPO_ROOT" rev-parse --short HEAD)}
export APERTUS_TRACK=${APERTUS_TRACK:-transformer-pp-350m-fineweb-edu-smoke}

exec bash "$SCRIPT_DIR/transformer-pp-350m-adamw.sbatch"
