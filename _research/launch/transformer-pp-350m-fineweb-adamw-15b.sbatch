#!/bin/bash
#
# Transformer++ 350M AdamW on the 15B-token FineWeb-Edu LLaMA-2 dataset.
# This wrapper pins the dataset and checkpoint defaults, then runs the
# reusable 350M AdamW launcher.
#
#SBATCH --job-name=transformer-pp-350m-fineweb-adamw-15b
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

REPO_DIR=/iopsstor/scratch/cscs/$USER/cler
cd "$REPO_DIR"
export SLURM_SUBMIT_DIR=$REPO_DIR

RUN_ID=transformer-pp-350m-fineweb-adamw-15b
STAMP=${STAMP:-$(date +%Y%m%d-%H%M)}
LOCAL_DATA_PREFIX=$REPO_DIR/_research/results/data/fineweb_edu/fineweb_edu_15b_llama2/fineweb_edu_15b_llama2_text_document
DEFAULT_DATA_PREFIX=${FINEWEB_15B_DATA_PREFIX:-$LOCAL_DATA_PREFIX}

export MEGATRON_DATA_PATH=${MEGATRON_DATA_PATH:-$DEFAULT_DATA_PREFIX}
if [ ! -f "${MEGATRON_DATA_PATH}.bin" ] || [ ! -f "${MEGATRON_DATA_PATH}.idx" ]; then
    echo "ERROR: compatible 15B FineWeb-Edu LLaMA-2 Megatron dataset not found at:" >&2
    echo "  $MEGATRON_DATA_PATH" >&2
    echo "Set FINEWEB_15B_DATA_PREFIX or MEGATRON_DATA_PATH to a matching prefix, or build the local one first:" >&2
    echo "  export LLAMA2_TOKENIZER_MODEL=/iopsstor/scratch/cscs/course_00206/llama2-tokenizer/tokenizer.model" >&2
    echo "  sbatch _research/data/convert_fineweb_edu.sbatch" >&2
    exit 1
fi

export LLAMA2_TOKENIZER_MODEL=${LLAMA2_TOKENIZER_MODEL:-/iopsstor/scratch/cscs/course_00206/llama2-tokenizer/tokenizer.model}
if [ ! -f "$LLAMA2_TOKENIZER_MODEL" ]; then
    echo "ERROR: tokenizer model not found: $LLAMA2_TOKENIZER_MODEL" >&2
    exit 1
fi

if [ -n "${TRAIN_ITERS:-}" ]; then
    echo "WARNING: TRAIN_ITERS=$TRAIN_ITERS overrides the default 15B TRAIN_SAMPLES target." >&2
fi

export EXP_NAME=${EXP_NAME:-$RUN_ID-$STAMP}
export APERTUS_TRACK=${APERTUS_TRACK:-$RUN_ID}
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-3662109}
export EVAL_INTERVAL=${EVAL_INTERVAL:-1907}
export EVAL_ITERS=${EVAL_ITERS:-10}
export SAVE_DIR=${SAVE_DIR:-$REPO_DIR/_research/results/checkpoints/$RUN_ID-ckpt-$STAMP}
export SAVE_INTERVAL=${SAVE_INTERVAL:-1907}
export CKPT_FORMAT=${CKPT_FORMAT:-torch}
export NO_SAVE_OPTIM=${NO_SAVE_OPTIM:-1}
export NO_SAVE_RNG=${NO_SAVE_RNG:-1}

exec bash "$REPO_DIR/_research/launch/transformer-pp-350m-adamw.sbatch"
