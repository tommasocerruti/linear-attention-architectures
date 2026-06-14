#!/bin/bash
# 2-node / 8-GPU multi-node bring-up smoke for the 1.3B AttnRes path (10 iters).
# RUN THIS FIRST (project hard rule): confirm it reaches training iterations and exits 0:0 before
# submitting any full 16-node run. We could not test multi-node from our account.
# EDIT: edit paths/W&B in transformer-pp-1p3b-fwe40b-env.sh, and `mkdir -p` the runs/ dir below.
#SBATCH --job-name=dbg-1p3b-attnres
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler-mega-cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler-mega-cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail
export EXIT_DURATION_IN_MINS=25
export TRAIN_ITERS=10            # overrides TRAIN_SAMPLES -> short smoke
export EVAL_INTERVAL=5
export EVAL_ITERS=1
export SAVE_INTERVAL=10
export WANDB_IS_SMOKE=1
export GBS=128                   # small global batch so 10 iters is quick on 8 GPUs
# Exercise the AttnRes + hybrid path (the most code); switch flags to smoke other variants.
export LINEAR_VARIANT=gdn_triton
export LINEAR_ATTENTION_FREQ=3
export ATTN_RES_ENABLED=1
export WANDB_ARCH=hybrid-gdn-triton-attnres-smoke
export EXP_NAME=dbg-1p3b-hybrid-attnres-smoke

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/transformer-pp-1p3b-fwe40b-env.sh"
