#!/bin/bash
# 1.3B / 40B-token PURE GDN (Triton/FLA) baseline, Muon. Baseline for the AttnRes/CLER family.
# EDIT: also edit paths/W&B in transformer-pp-1p3b-fwe40b-env.sh, and `mkdir -p` the runs/ dir below.
#SBATCH --job-name=1p3b-fwe40b-pure-gdn
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=14:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/linear-attention-architectures/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/linear-attention-architectures/_research/results/runs/%x-%j.log
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail
export EXIT_DURATION_IN_MINS=800
export LINEAR_VARIANT=gdn_triton
export WANDB_ARCH=pure-gdn-triton
export EXP_NAME=pure-gdn-triton-1p3b-fwe40b-muon

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/transformer-pp-1p3b-fwe40b-env.sh"
