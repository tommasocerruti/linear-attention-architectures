#!/bin/bash
# 1.3B / 40B-token HYBRID GDN + CLER-H (hidden-stream error routing, LOW-RANK projection rank=16),
# Muon. CLER-H was the first CLER variant to beat the GDN baseline at 350M/1B (-0.0065); this scales
# the low-rank (param-light) version. Compare to transformer-pp-1p3b-fwe40b-hybrid-gdn-muon.sbatch.
# Also edit paths/W&B/account in transformer-pp-1p3b-fwe40b-env.sh, and `mkdir -p` the runs/ dir.
#SBATCH --job-name=1p3b-fwe40b-cler-hidden-route
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=14:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler-mega-cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler-mega-cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail
export EXIT_DURATION_IN_MINS=800
export LINEAR_VARIANT=cler_gdn_triton
export LINEAR_ATTENTION_FREQ=3        # hybrid: 2 GDN : 1 SDPA (CLER-H routes the GDN write errors)
export CLER_HIDDEN_ROUTING=1
export CLER_HIDDEN_RANK=16
export WANDB_ARCH=hybrid-cler-gdn-hidden-route-lowrank
export EXP_NAME=hybrid-cler-gdn-hidden-route-lowrank-1p3b-fwe40b-muon

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/transformer-pp-1p3b-fwe40b-env.sh"
