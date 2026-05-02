#!/bin/bash
#
# PyTorch Gated DeltaNet 350M Muon run on FineWeb-Edu/LLaMA-2.
#
#SBATCH --job-name=transformer-pp-350m-gdn-pytorch-muon
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

export LINEAR_VARIANT=gdn_pytorch
export WANDB_ARCH=${WANDB_ARCH:-gdn-pytorch}
export APERTUS_FEATURE=${APERTUS_FEATURE:-gdn-pytorch-muon}
export APERTUS_TRACK=${APERTUS_TRACK:-transformer-pp-350m-gdn-pytorch-muon}

exec bash "$SCRIPT_DIR/transformer-pp-350m-linear-muon.sbatch"
