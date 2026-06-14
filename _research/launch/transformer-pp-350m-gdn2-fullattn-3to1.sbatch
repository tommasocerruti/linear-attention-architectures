#!/bin/bash
#
# Hybrid Gated DeltaNet2 ~350M baseline on 1 GH200 node (4 GPUs).
# 3:1 GDN2:standard full self-attention layer pattern.
#
# In the experimental GPT path, linear_attention_freq=4 gives
# [GDN2, GDN2, GDN2, full-attention] repeated. The full-attention layers use
# the repo's normal Transformer self-attention spec with attention-backend auto.
#
# Requires MEGATRON_DATA_PATH to point at a FineWeb-Edu Megatron prefix and
# LLAMA2_TOKENIZER_MODEL to point at a local LLaMA-2 tokenizer.model.
#
#SBATCH --job-name=transformer-pp-350m-gdn2-fullattn-3to1
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

RUN_ID=transformer-pp-350m-gdn2-fullattn-3to1
GIT_SHA=$(git -C "$CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)

export LINEAR_ATTENTION_FREQ=${LINEAR_ATTENTION_FREQ:-4}
export EXP_NAME=${EXP_NAME:-$RUN_ID-$GIT_SHA}
export APERTUS_TRACK=${APERTUS_TRACK:-$RUN_ID}

exec bash "$SCRIPT_DIR/transformer-pp-350m-gdn2.sbatch"
