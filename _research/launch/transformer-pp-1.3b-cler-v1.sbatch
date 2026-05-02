#!/bin/bash
#
# Experimental CLER v1 PyTorch Gated DeltaNet ~1.3B run on 8 GH200 nodes (32 GPUs).
# This keeps the same CLER/GDN code path as the 350M launcher and scales the model
# shape to the repo's 1.3B Transformer++ preset.
#
# Required environment:
#   MEGATRON_DATA_PATH=/path/to/fineweb_edu_100b_llama2_text_document
#   LLAMA2_TOKENIZER_MODEL=/path/to/tokenizer.model
#
# Run the 350M CLER launcher first before using this scaling target.
#
#SBATCH --job-name=transformer-pp-1.3b-cler-v1
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=8
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

GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)

export CLER_ENABLED=1
export CLER_GAMMA_INIT=${CLER_GAMMA_INIT:-0.0}
export CLER_DETACH_RESIDUAL=${CLER_DETACH_RESIDUAL:-0}

export NUM_LAYERS=${NUM_LAYERS:-24}
export HIDDEN=${HIDDEN:-2048}
export FFN_HIDDEN=${FFN_HIDDEN:-5632}
export NUM_HEADS=${NUM_HEADS:-32}
export NUM_KV_HEADS=${NUM_KV_HEADS:-8}
export MBS=${MBS:-2}
export GBS=${GBS:-128}
export SEQ_LEN=${SEQ_LEN:-4096}
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-24414063}
export EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-1430}
export LR=${LR:-2e-4}
export MIN_LR=${MIN_LR:-2e-5}
export LINEAR_ATTENTION_FREQ=${LINEAR_ATTENTION_FREQ:-3}
export LINEAR_KEY_HEAD_DIM=${LINEAR_KEY_HEAD_DIM:-64}
export LINEAR_VALUE_HEAD_DIM=${LINEAR_VALUE_HEAD_DIM:-64}
export LINEAR_NUM_KEY_HEADS=${LINEAR_NUM_KEY_HEADS:-8}
export LINEAR_NUM_VALUE_HEADS=${LINEAR_NUM_VALUE_HEADS:-16}

export EXP_NAME=${EXP_NAME:-transformer-pp-1.3b-cler-v1-$GIT_SHA}
export APERTUS_FEATURE=${APERTUS_FEATURE:-cler-v1-gdn-pytorch}
export APERTUS_TRACK=${APERTUS_TRACK:-transformer-pp-1.3b-cler-v1}
export WANDB_ENTITY=${WANDB_ENTITY:-cler}
export WANDB_PROJECT=${WANDB_PROJECT:-clerv1-runs}
export WANDB_GROUP=${WANDB_GROUP:-clerv1}
export WANDB_ARCH=${WANDB_ARCH:-cler-v1-gdn}
export WANDB_MODEL_SIZE=${WANDB_MODEL_SIZE:-1.3B}

exec bash "$SCRIPT_DIR/transformer-pp-350m-gdn-pytorch.sbatch"
