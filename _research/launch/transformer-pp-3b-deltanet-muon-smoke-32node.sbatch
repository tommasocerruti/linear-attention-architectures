#!/bin/bash
#
# 32-node timing smoke for 3B DeltaNet Muon hybrid.
# Purpose: measure real iter time at scale before committing to full run.
# 32 nodes (128 GPUs), normal partition, MBS=4, GBS=2048.
# Each GPU: 2048/128=16 samples, 16/4=4 grad-acc steps — tests overlap benefit.
#
#SBATCH --job-name=dn-muon-3b-smoke-32node
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

export TRAIN_ITERS=50
export LR_WARMUP_ITERS=5
export LR_DECAY_ITERS=50
export LR_WSD_DECAY_ITERS=10
export MBS=4
export GBS=2048
export WANDB_IS_SMOKE=1
export CKPT_FORMAT=torch

REPO_DIR=/iopsstor/scratch/cscs/$USER/cler
cd "$REPO_DIR"

exec bash "$REPO_DIR/_research/launch/transformer-pp-3b-deltanet-muon.sbatch"
