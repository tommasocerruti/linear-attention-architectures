#!/bin/bash
#
# 62B-token FineWeb-Edu prefix using dataset token_count and Megatron preprocessing.
# Normal partition + detailed timing for each step.
#
#SBATCH --job-name=convert-fineweb-edu-62b-tc
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=6:00:00
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
cd "$WORKDIR"

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
FINEWEB_EDU_TOKENS=${FINEWEB_EDU_TOKENS:-62000000000}
FINEWEB_EDU_WORKERS=${FINEWEB_EDU_WORKERS:-64}
FINEWEB_EDU_MAX_DOCUMENTS=${FINEWEB_EDU_MAX_DOCUMENTS:-}
FINEWEB_EDU_OUTPUT_ROOT=${FINEWEB_EDU_OUTPUT_ROOT:-$WORKDIR/_research/results/data/fineweb_edu}
FINEWEB_EDU_TOKENIZER_MODEL=${FINEWEB_EDU_TOKENIZER_MODEL:-${LLAMA2_TOKENIZER_MODEL:-}}
FINEWEB_EDU_TOKEN_COUNT_KEY=${FINEWEB_EDU_TOKEN_COUNT_KEY:-token_count}

if [ -z "$FINEWEB_EDU_TOKENIZER_MODEL" ]; then
    echo "ERROR: set LLAMA2_TOKENIZER_MODEL or FINEWEB_EDU_TOKENIZER_MODEL to a local LLaMA-2 tokenizer.model path." >&2
    exit 1
fi
if [ ! -f "$FINEWEB_EDU_TOKENIZER_MODEL" ]; then
    echo "ERROR: tokenizer model not found: $FINEWEB_EDU_TOKENIZER_MODEL" >&2
    exit 1
fi

case "$FINEWEB_EDU_TOKENS" in
    1000000000) DEFAULT_FINEWEB_EDU_NAME=fineweb_edu_1b_llama2_tc ;;
    15000000000) DEFAULT_FINEWEB_EDU_NAME=fineweb_edu_15b_llama2_tc ;;
    62000000000) DEFAULT_FINEWEB_EDU_NAME=fineweb_edu_62b_llama2_tc ;;
    100000000000) DEFAULT_FINEWEB_EDU_NAME=fineweb_edu_100b_llama2_tc ;;
    *) DEFAULT_FINEWEB_EDU_NAME=fineweb_edu_${FINEWEB_EDU_TOKENS}_llama2_tc ;;
esac

FINEWEB_EDU_NAME=${FINEWEB_EDU_NAME:-$DEFAULT_FINEWEB_EDU_NAME}
FINEWEB_EDU_DIR=$FINEWEB_EDU_OUTPUT_ROOT/$FINEWEB_EDU_NAME
FINEWEB_EDU_JSONL=$FINEWEB_EDU_DIR/$FINEWEB_EDU_NAME.jsonl
FINEWEB_EDU_PREFIX=$FINEWEB_EDU_DIR/$FINEWEB_EDU_NAME
FINEWEB_EDU_DATA_PATH=${FINEWEB_EDU_PREFIX}_text_document
TIMING_LOG=$FINEWEB_EDU_DIR/$FINEWEB_EDU_NAME.timings.log

mkdir -p "$FINEWEB_EDU_DIR"

: > "$TIMING_LOG"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log_timing() {
    local step=$1
    local start=$2
    local end=$3
    local elapsed=$((end - start))
    printf "[%s] %s: %s seconds\n" "$(timestamp)" "$step" "$elapsed" | tee -a "$TIMING_LOG"
}

PREPARE_EXTRA_ARGS=()
if [ -n "$FINEWEB_EDU_MAX_DOCUMENTS" ]; then
    PREPARE_EXTRA_ARGS+=(--max-documents "$FINEWEB_EDU_MAX_DOCUMENTS")
fi

STEP_START=$(date +%s)
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=$WORKDIR/_research/launch/alps3.toml \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 120 \
    bash -lc "set -euo pipefail; \
        cd $WORKDIR; \
        export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR\${PYTHONPATH:+:\$PYTHONPATH}; \
        export HF_HOME=$HF_HOME; \
        export HF_DATASETS_CACHE=$HF_DATASETS_CACHE; \
        export TMPDIR=$TMPDIR; \
        export OMP_NUM_THREADS=$OMP_NUM_THREADS; \
        bash $WORKDIR/_research/launch/install_python_deps.sh $PACKAGE_DIR"
STEP_END=$(date +%s)
log_timing "install_deps" $STEP_START $STEP_END

STEP_START=$(date +%s)
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=$WORKDIR/_research/launch/alps3.toml \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 120 \
    bash -lc "set -euo pipefail; \
        cd $WORKDIR; \
        export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR\${PYTHONPATH:+:\$PYTHONPATH}; \
        export HF_HOME=$HF_HOME; \
        export HF_DATASETS_CACHE=$HF_DATASETS_CACHE; \
        export TMPDIR=$TMPDIR; \
        export OMP_NUM_THREADS=$OMP_NUM_THREADS; \
        set +e; \
        python3 $WORKDIR/_research/data/prepare_fineweb_edu.py \
            --dataset $FINEWEB_EDU_DATASET \
            --split $FINEWEB_EDU_SPLIT \
            --target-tokens $FINEWEB_EDU_TOKENS \
            --token-count-key $FINEWEB_EDU_TOKEN_COUNT_KEY \
            --sentencepiece-tokenizer-model $FINEWEB_EDU_TOKENIZER_MODEL \
            --output $FINEWEB_EDU_JSONL \
            ${PREPARE_EXTRA_ARGS[*]}; \
        PREPARE_STATUS=\$?; \
        set -e; \
        if [ \$PREPARE_STATUS -ne 0 ]; then \
            python3 -c 'import json, pathlib, sys; manifest = pathlib.Path(sys.argv[1]); data = json.loads(manifest.read_text()) if manifest.exists() else {}; sys.exit(0 if data.get(\"status\") == \"complete\" else 1)' ${FINEWEB_EDU_JSONL}.manifest.json || exit \$PREPARE_STATUS; \
            echo 'prepare_fineweb_edu.py exited non-zero after writing a complete manifest; continuing to preprocessing.'; \
        fi"
STEP_END=$(date +%s)
log_timing "prepare_jsonl" $STEP_START $STEP_END

STEP_START=$(date +%s)
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=$WORKDIR/_research/launch/alps3.toml \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 120 \
    bash -lc "set -euo pipefail; \
        cd $WORKDIR; \
        export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR\${PYTHONPATH:+:\$PYTHONPATH}; \
        export HF_HOME=$HF_HOME; \
        export HF_DATASETS_CACHE=$HF_DATASETS_CACHE; \
        export TMPDIR=$TMPDIR; \
        export OMP_NUM_THREADS=$OMP_NUM_THREADS; \
        python3 $WORKDIR/tools/preprocess_data.py \
            --input $FINEWEB_EDU_JSONL \
            --output-prefix $FINEWEB_EDU_PREFIX \
            --tokenizer-type Llama2Tokenizer \
            --tokenizer-model $FINEWEB_EDU_TOKENIZER_MODEL \
            --append-eod \
            --workers $FINEWEB_EDU_WORKERS \
            --log-interval 10000"
STEP_END=$(date +%s)
log_timing "preprocess_megatron" $STEP_START $STEP_END

echo "FineWeb-Edu Megatron prefix:"
echo "export MEGATRON_DATA_PATH=$FINEWEB_EDU_DATA_PATH"
echo "Timing log: $TIMING_LOG"
echo "END TIME: $(date)"
