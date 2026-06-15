#!/bin/bash
# 1.3B / 40B-token HYBRID GDN/SDPA (linear_attention_freq=3) + Full Attention Residuals, Muon.
# RECOMMENDED AttnRes test: Kimi Linear (where AttnRes helped) is a hybrid linear+full-attention
# stack, so this is the faithful analog. Compare to transformer-pp-1p3b-fwe40b-hybrid-gdn-muon.sbatch.
# EDIT: also edit paths/W&B in transformer-pp-1p3b-fwe40b-env.sh, and `mkdir -p` the runs/ dir below.
#SBATCH --job-name=1p3b-fwe40b-hybrid-gdn-attnres
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
export LINEAR_ATTENTION_FREQ=3        # hybrid: 2 GDN : 1 SDPA
export ATTN_RES_ENABLED=1
export WANDB_ARCH=hybrid-gdn-triton-attnres
export EXP_NAME=hybrid-gdn-triton-attnres-1p3b-fwe40b-muon

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/transformer-pp-1p3b-fwe40b-env.sh"
