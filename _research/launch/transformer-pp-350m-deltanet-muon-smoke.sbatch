#!/bin/bash
#
# Short NorMuon runtime smoke for the DeltaNet hybrid 350M model.
# Reuses the full deltanet-muon launcher with a small stopping condition.
#
#SBATCH --job-name=transformer-pp-350m-deltanet-muon-smoke
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

REPO_ROOT=/iopsstor/scratch/cscs/$USER/cler
SCRIPT_DIR=$REPO_ROOT/_research/launch
cd "$REPO_ROOT"

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
GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
export EXP_NAME=${EXP_NAME:-transformer-pp-350m-deltanet-muon-smoke-$GIT_SHA}
export APERTUS_TRACK=${APERTUS_TRACK:-transformer-pp-350m-deltanet-muon-smoke}

exec bash "$SCRIPT_DIR/transformer-pp-350m-deltanet-muon.sbatch"
