#!/bin/bash
#
# CLER v1 PyTorch Gated DeltaNet 350M Muon run on FineWeb-Edu/LLaMA-2.
#
#SBATCH --job-name=transformer-pp-350m-cler-v1-muon
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

REPO_ROOT=${CLER_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/iopsstor/scratch/cscs/$USER/cler}}
if [ ! -f "$REPO_ROOT/pretrain_gpt.py" ]; then
    REPO_ROOT=/iopsstor/scratch/cscs/$USER/cler
fi
SCRIPT_DIR=$REPO_ROOT/_research/launch
cd "$REPO_ROOT"

export LINEAR_VARIANT=cler
export CLER_ENABLED=1
export CLER_GAMMA_INIT=${CLER_GAMMA_INIT:-0.0}
export CLER_DETACH_RESIDUAL=${CLER_DETACH_RESIDUAL:-0}
export WANDB_ARCH=${WANDB_ARCH:-cler-v1-gdn-pytorch}
export WANDB_GROUP=${WANDB_GROUP:-clerv1}
export APERTUS_FEATURE=${APERTUS_FEATURE:-cler-v1-gdn-pytorch-muon}
export APERTUS_TRACK=${APERTUS_TRACK:-transformer-pp-350m-cler-v1-muon}

exec bash "$SCRIPT_DIR/transformer-pp-350m-linear-muon.sbatch"
