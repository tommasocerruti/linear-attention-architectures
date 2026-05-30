#!/bin/bash
# 1.3B / 40B-token PURE GDN + CLER dense-softmax routing (AttnRes-style depth routing over
# residuals; per-head gamma), Muon.
# EDIT: also edit paths/W&B in transformer-pp-1p3b-fwe40b-env.sh, and `mkdir -p` the runs/ dir below.
#SBATCH --job-name=1p3b-fwe40b-cler-densesoftmax
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=15:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler-mega-cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler-mega-cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail
export EXIT_DURATION_IN_MINS=860
export LINEAR_VARIANT=cler_gdn_triton
export CLER_ROUTING_MODE=dense_softmax
export CLER_GAMMA_MODE=head
export WANDB_ARCH=pure-cler-gdn-densesoftmax
export EXP_NAME=pure-cler-gdn-densesoftmax-1p3b-fwe40b-muon

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/transformer-pp-1p3b-fwe40b-env.sh"
