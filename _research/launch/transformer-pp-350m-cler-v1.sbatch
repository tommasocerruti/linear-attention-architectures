#!/bin/bash
#
# CLER v1 PyTorch Gated DeltaNet ~350M run on 1 GH200 node (4 GPUs).
# Uses the FineWeb-Edu/LLaMA-2 PyTorch GDN launcher with CLER enabled.
#
# Required environment:
#   MEGATRON_DATA_PATH=/path/to/fineweb_edu_15b_llama2_text_document
#   LLAMA2_TOKENIZER_MODEL=/path/to/tokenizer.model
#
# Optional smoke override:
#   TRAIN_ITERS=50 MBS=2 GBS=8 EXIT_DURATION_IN_MINS=25 EVAL_INTERVAL=25 EVAL_ITERS=1 \
#     sbatch --partition=debug --time=00:45:00 _research/launch/transformer-pp-350m-cler-v1.sbatch
#
#SBATCH --job-name=transformer-pp-350m-cler-v1
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

export NUM_LAYERS=${NUM_LAYERS:-20}
export HIDDEN=${HIDDEN:-1024}
export FFN_HIDDEN=${FFN_HIDDEN:-2816}
export NUM_HEADS=${NUM_HEADS:-16}
export NUM_KV_HEADS=${NUM_KV_HEADS:-4}
export MBS=${MBS:-2}
export GBS=${GBS:-128}
export SEQ_LEN=${SEQ_LEN:-4096}
export LINEAR_ATTENTION_FREQ=${LINEAR_ATTENTION_FREQ:-3}
export LINEAR_KEY_HEAD_DIM=${LINEAR_KEY_HEAD_DIM:-64}
export LINEAR_VALUE_HEAD_DIM=${LINEAR_VALUE_HEAD_DIM:-64}
export LINEAR_NUM_KEY_HEADS=${LINEAR_NUM_KEY_HEADS:-4}
export LINEAR_NUM_VALUE_HEADS=${LINEAR_NUM_VALUE_HEADS:-8}

export EXP_NAME=${EXP_NAME:-transformer-pp-350m-cler-v1-$GIT_SHA}
export APERTUS_FEATURE=${APERTUS_FEATURE:-cler-v1-gdn-pytorch}
export APERTUS_TRACK=${APERTUS_TRACK:-transformer-pp-350m-cler-v1}
export WANDB_ENTITY=${WANDB_ENTITY:-cler}
export WANDB_PROJECT=${WANDB_PROJECT:-clerv1-runs}
export WANDB_GROUP=${WANDB_GROUP:-clerv1}
export WANDB_ARCH=${WANDB_ARCH:-cler-v1-gdn}
export WANDB_MODEL_SIZE=${WANDB_MODEL_SIZE:-350M}

exec bash "$SCRIPT_DIR/transformer-pp-350m-gdn-pytorch.sbatch"
