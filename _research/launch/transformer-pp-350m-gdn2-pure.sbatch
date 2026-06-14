#!/bin/bash
#
# Pure Gated DeltaNet2 ~350M baseline on 1 GH200 node (4 GPUs).
# Reuses the GDN2 launcher but forces every transformer layer to use GDN2.
#
# Requires MEGATRON_DATA_PATH to point at a FineWeb-Edu Megatron prefix and
# LLAMA2_TOKENIZER_MODEL to point at a local LLaMA-2 tokenizer.model.
#
#SBATCH --job-name=transformer-pp-350m-gdn2-pure
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

RUN_ID=transformer-pp-350m-gdn2-pure
GIT_SHA=$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)

export NUM_LAYERS=${NUM_LAYERS:-20}
export LINEAR_ATTENTION_FREQ=${LINEAR_ATTENTION_FREQ:-"[1]*$NUM_LAYERS"}
export EXP_NAME=${EXP_NAME:-$RUN_ID-$GIT_SHA}
export APERTUS_TRACK=${APERTUS_TRACK:-$RUN_ID}

exec bash "$SCRIPT_DIR/transformer-pp-350m-gdn2.sbatch"
