#!/bin/bash
#
# Pure DeltaNet ~1.3B Muon on 8 GH200 nodes (32 GPUs).
# All 24 layers use DeltaNet (no SDPA hybrid).
# 40B tokens, GBS=512, MBS=8.
#
#SBATCH --job-name=transformer-pp-1.3b-deltanet-muon-pure
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

export LINEAR_ATTENTION_FREQ="[1]*24"
export WANDB_PROJECT=${WANDB_PROJECT:-cler-runs-1.3B}

REPO_DIR=/iopsstor/scratch/cscs/$USER/cler
cd "$REPO_DIR"

exec bash "$REPO_DIR/_research/launch/transformer-pp-1.3b-deltanet-muon.sbatch"
