#!/bin/bash
# 1.3B / 40B-token PURE GDN + Full Attention Residuals, Muon.
# NOTE: at 350M/1B, PURE AttnRes was clearly WORSE than pure GDN (+0.068 val); HYBRID was neutral.
# Kimi Linear (where AttnRes helped) is a HYBRID linear+full-attention stack, so the hybrid wrapper
# (transformer-pp-1p3b-fwe40b-hybrid-gdn-attnres-muon.sbatch) is the more faithful analog. This pure run is
# included because it was on your list; compare it to transformer-pp-1p3b-fwe40b-pure-gdn-muon.sbatch.
# EDIT: also edit paths/W&B in transformer-pp-1p3b-fwe40b-env.sh, and `mkdir -p` the runs/ dir below.
#SBATCH --job-name=1p3b-fwe40b-pure-gdn-attnres
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=23:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/linear-attention-architectures/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/linear-attention-architectures/_research/results/runs/%x-%j.log
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail
export EXIT_DURATION_IN_MINS=1340
export LINEAR_VARIANT=gdn_triton
export ATTN_RES_ENABLED=1
export WANDB_ARCH=pure-gdn-triton-attnres
export EXP_NAME=pure-gdn-triton-attnres-1p3b-fwe40b-muon

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/transformer-pp-1p3b-fwe40b-env.sh"
