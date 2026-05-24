#!/bin/bash
#
# Hybrid Gated DeltaNet2 ~350M baseline on 1 GH200 node (4 GPUs).
# 2:1 GDN2:standard attention layer pattern, using NorMuon
# (adaptive_muon + normuon). Keeps external GDN2 kernels outside CLER.
# --overlap-param-gather intentionally omitted; it breaks Muon.
#
# Requires MEGATRON_DATA_PATH to point at a FineWeb-Edu Megatron prefix and
# LLAMA2_TOKENIZER_MODEL to point at a local LLaMA-2 tokenizer.model.
#
#SBATCH --job-name=transformer-pp-350m-gdn2-muon
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

CODE_DIR=${CODE_DIR:-/users/$USER/cler}
SCRIPT_DIR=$CODE_DIR/_research/launch
if [ ! -f "$CODE_DIR/pretrain_gpt.py" ] || [ ! -d "$CODE_DIR/megatron" ]; then
    echo "ERROR: CLER code checkout not found at $CODE_DIR" >&2
    exit 1
fi
cd "$CODE_DIR"

RUN_ID=transformer-pp-350m-gdn2-muon
GIT_SHA=$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)

export OPTIMIZER=${OPTIMIZER:-adaptive_muon}
export LR=${LR:-3.6e-4}
export MIN_LR=${MIN_LR:-3.6e-5}
export MUON_SCALE_MODE=${MUON_SCALE_MODE:-spectral}
export MUON_MOMENTUM=${MUON_MOMENTUM:-0.95}
export MUON_SCALAR_LR=${MUON_SCALAR_LR:-1.5e-3}
export MUON_SCALAR_WEIGHT_DECAY=${MUON_SCALAR_WEIGHT_DECAY:-0.0}
export ADAPTIVE_MUON_MOMENT2_METHOD=${ADAPTIVE_MUON_MOMENT2_METHOD:-normuon}
export MUON_NO_SPLIT_QKV=${MUON_NO_SPLIT_QKV:-1}
export EXP_NAME=${EXP_NAME:-$RUN_ID-$GIT_SHA}
export APERTUS_TRACK=${APERTUS_TRACK:-$RUN_ID}

exec bash "$SCRIPT_DIR/transformer-pp-350m-gdn2.sbatch"
