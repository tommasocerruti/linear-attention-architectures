#!/bin/bash
#
# Pure DeltaNet 350M AdamW on the 15B-token FineWeb-Edu LLaMA-2 dataset.
# This wrapper reuses the 15B DeltaNet AdamW launcher but forces every layer to use
# linear attention rather than the default hybrid LA/SDPA pattern.
#
#SBATCH --job-name=transformer-pp-350m-deltanet-adamw-15b-pure-deltarule
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

REPO_DIR=/iopsstor/scratch/cscs/$USER/cler
cd "$REPO_DIR"
export SLURM_SUBMIT_DIR=$REPO_DIR

RUN_ID=transformer-pp-350m-deltanet-adamw-15b-pure-deltarule
STAMP=${STAMP:-$(date +%Y%m%d-%H%M)}

export LINEAR_ATTENTION_FREQ=${LINEAR_ATTENTION_FREQ:-"[1]*24"}
export EXP_NAME=${EXP_NAME:-$RUN_ID-$STAMP}
export APERTUS_TRACK=${APERTUS_TRACK:-$RUN_ID}
export SAVE_DIR=${SAVE_DIR:-$REPO_DIR/_research/results/checkpoints/$RUN_ID-ckpt-$STAMP}

exec bash "$REPO_DIR/_research/launch/transformer-pp-350m-deltanet-adamw-15b.sbatch"
