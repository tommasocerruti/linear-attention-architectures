#!/bin/bash
#
# Smoke test for 3B DeltaNet Muon hybrid.
# 4 nodes (16 GPUs), debug partition — mirrors full run config as closely as possible.
# GBS=512, MBS=4: 8 samples/GPU, 2 grad-acc steps.
# MBS=4 instead of full run's MBS=2 to stress-test memory (est. ~83 GB peak).
#
#SBATCH --job-name=dn-muon-3b-smoke
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:20:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

export TRAIN_ITERS=50
export LR_WARMUP_ITERS=5
export LR_DECAY_ITERS=50
export LR_WSD_DECAY_ITERS=10
export MBS=8
export GBS=512
export WANDB_IS_SMOKE=1
export CKPT_FORMAT=torch

REPO_DIR=/iopsstor/scratch/cscs/$USER/cler
cd "$REPO_DIR"

exec bash "$REPO_DIR/_research/launch/transformer-pp-3b-deltanet-muon.sbatch"
