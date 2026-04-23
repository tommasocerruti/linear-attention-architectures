#!/bin/bash
#
# Download a deterministic FineWeb-Edu prefix and convert it to Megatron
# binary format with a LLaMA-2 SentencePiece tokenizer.
#
# Defaults:
#   15B tokens from HuggingFaceFW/fineweb_edu_100BT-shuffled
#   output under _research/results/data/fineweb_edu/
#
# To build the 100B prefix:
#   FINEWEB_EDU_TOKENS=100000000000 FINEWEB_EDU_NAME=fineweb_edu_100b_llama2 \
#     sbatch _research/data/convert_fineweb_edu.sbatch
#
#SBATCH --job-name=convert-fineweb-edu
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail
echo "START TIME: $(date)"

REPO_DIR=/iopsstor/scratch/cscs/$USER/cler
WORKDIR=${SLURM_SUBMIT_DIR:-$REPO_DIR}
cd $WORKDIR

PACKAGE_DIR=$REPO_DIR/_research/packages
HF_CACHE_DIR=$REPO_DIR/cache/huggingface
TMP_CACHE_DIR=$REPO_DIR/cache/tmp

export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=$HF_CACHE_DIR
export HF_DATASETS_CACHE=$HF_CACHE_DIR/datasets
export TMPDIR=$TMP_CACHE_DIR
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p "$PACKAGE_DIR" "$HF_HOME" "$TMP_CACHE_DIR" _research/results/runs

FINEWEB_EDU_DATASET=${FINEWEB_EDU_DATASET:-HuggingFaceFW/fineweb_edu_100BT-shuffled}
FINEWEB_EDU_SPLIT=${FINEWEB_EDU_SPLIT:-train}
FINEWEB_EDU_TOKENS=${FINEWEB_EDU_TOKENS:-15000000000}
FINEWEB_EDU_WORKERS=${FINEWEB_EDU_WORKERS:-64}
FINEWEB_EDU_MAX_DOCUMENTS=${FINEWEB_EDU_MAX_DOCUMENTS:-}
FINEWEB_EDU_OUTPUT_ROOT=${FINEWEB_EDU_OUTPUT_ROOT:-$WORKDIR/_research/results/data/fineweb_edu}
FINEWEB_EDU_TOKENIZER_MODEL=${FINEWEB_EDU_TOKENIZER_MODEL:-${LLAMA2_TOKENIZER_MODEL:-}}

if [ -z "$FINEWEB_EDU_TOKENIZER_MODEL" ]; then
    echo "ERROR: set LLAMA2_TOKENIZER_MODEL or FINEWEB_EDU_TOKENIZER_MODEL to a local LLaMA-2 tokenizer.model path." >&2
    exit 1
fi
if [ ! -f "$FINEWEB_EDU_TOKENIZER_MODEL" ]; then
    echo "ERROR: tokenizer model not found: $FINEWEB_EDU_TOKENIZER_MODEL" >&2
    exit 1
fi

case "$FINEWEB_EDU_TOKENS" in
    15000000000) DEFAULT_FINEWEB_EDU_NAME=fineweb_edu_15b_llama2 ;;
    100000000000) DEFAULT_FINEWEB_EDU_NAME=fineweb_edu_100b_llama2 ;;
    *) DEFAULT_FINEWEB_EDU_NAME=fineweb_edu_${FINEWEB_EDU_TOKENS}_llama2 ;;
esac

FINEWEB_EDU_NAME=${FINEWEB_EDU_NAME:-$DEFAULT_FINEWEB_EDU_NAME}
FINEWEB_EDU_DIR=$FINEWEB_EDU_OUTPUT_ROOT/$FINEWEB_EDU_NAME
FINEWEB_EDU_JSONL=$FINEWEB_EDU_DIR/$FINEWEB_EDU_NAME.jsonl
FINEWEB_EDU_PREFIX=$FINEWEB_EDU_DIR/$FINEWEB_EDU_NAME
FINEWEB_EDU_DATA_PATH=${FINEWEB_EDU_PREFIX}_text_document

mkdir -p "$FINEWEB_EDU_DIR"

bash "$WORKDIR/_research/launch/install_python_deps.sh" "$PACKAGE_DIR"

PREPARE_ARGS=(
    --dataset "$FINEWEB_EDU_DATASET"
    --split "$FINEWEB_EDU_SPLIT"
    --target-tokens "$FINEWEB_EDU_TOKENS"
    --token-count-key ""
    --sentencepiece-tokenizer-model "$FINEWEB_EDU_TOKENIZER_MODEL"
    --output "$FINEWEB_EDU_JSONL"
)
if [ -n "$FINEWEB_EDU_MAX_DOCUMENTS" ]; then
    PREPARE_ARGS+=(--max-documents "$FINEWEB_EDU_MAX_DOCUMENTS")
fi

python3 "$WORKDIR/_research/data/prepare_fineweb_edu.py" \
    "${PREPARE_ARGS[@]}"

python3 "$WORKDIR/tools/preprocess_data.py" \
    --input "$FINEWEB_EDU_JSONL" \
    --output-prefix "$FINEWEB_EDU_PREFIX" \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model "$FINEWEB_EDU_TOKENIZER_MODEL" \
    --append-eod \
    --workers "$FINEWEB_EDU_WORKERS" \
    --log-interval 10000

echo "FineWeb-Edu Megatron prefix:"
echo "export MEGATRON_DATA_PATH=$FINEWEB_EDU_DATA_PATH"
echo "END TIME: $(date)"
