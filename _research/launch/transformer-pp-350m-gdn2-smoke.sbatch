#!/bin/bash
#
# Short Gated DeltaNet2 runtime smoke for the FineWeb-Edu/LLaMA-2 path.
# Reuses the full 350M GDN2 launcher with a small stopping condition.
#
#SBATCH --job-name=transformer-pp-350m-gdn2-smoke
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

CODE_DIR=${CODE_DIR:-/users/$USER/cler}
SCRIPT_DIR=$CODE_DIR/_research/launch
if [ ! -f "$CODE_DIR/pretrain_gpt.py" ] || [ ! -d "$CODE_DIR/megatron" ]; then
    echo "ERROR: CLER code checkout not found at $CODE_DIR" >&2
    exit 1
fi
cd "$CODE_DIR"

export TRAIN_ITERS=${TRAIN_ITERS:-50}
export EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-25}
export EVAL_INTERVAL=${EVAL_INTERVAL:-25}
export EVAL_ITERS=${EVAL_ITERS:-1}
export SEQ_LEN=${SEQ_LEN:-4096}
export MBS=${MBS:-16}
export GBS=${GBS:-128}
export RECOMPUTE=${RECOMPUTE:-0}
export CP=${CP:-1}
export WANDB_IS_SMOKE=1
GIT_SHA=$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)
export EXP_NAME=${EXP_NAME:-transformer-pp-350m-gdn2-smoke-$GIT_SHA}
export APERTUS_TRACK=${APERTUS_TRACK:-transformer-pp-350m-gdn2-smoke}

exec bash "$SCRIPT_DIR/transformer-pp-350m-gdn2.sbatch"
