#!/bin/bash
#
# Prepare a small FineWeb-Edu tail slice and submit native-Megatron lm_eval
# perplexity against the local completions server.

set -euo pipefail

: "${CKPT:?set CKPT to a native Megatron checkpoint root dir}"
: "${LLAMA2_TOKENIZER_MODEL:?set LLAMA2_TOKENIZER_MODEL to tokenizer.model}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKDIR=${WORKDIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}
cd "$WORKDIR"

FINEWEB_EDU_SOURCE_JSONL=${FINEWEB_EDU_SOURCE_JSONL:-/iopsstor/scratch/cscs/tr1eder/cler/_research/results/data/fineweb_edu/fineweb_edu_15b_llama2_tc/fineweb_edu_15b_llama2_tc.jsonl}
FINEWEB_EDU_EVAL_DOCS=${FINEWEB_EDU_EVAL_DOCS:-10000}
FINEWEB_EDU_EVAL_OUTPUT=${FINEWEB_EDU_EVAL_OUTPUT:-$WORKDIR/_research/results/eval/fineweb_edu/fineweb_edu_15b_tail_${FINEWEB_EDU_EVAL_DOCS}.jsonl}
FINEWEB_EDU_TASK_NAME=${FINEWEB_EDU_TASK_NAME:-fineweb_edu_15b_tail}
FINEWEB_EDU_TASK_DIR=${FINEWEB_EDU_TASK_DIR:-$WORKDIR/_research/results/eval/fineweb_edu/tasks-${FINEWEB_EDU_EVAL_DOCS}}

python3 "$WORKDIR/_research/eval/prepare_fineweb_edu_eval_slice.py" \
    --source-jsonl "$FINEWEB_EDU_SOURCE_JSONL" \
    --output "$FINEWEB_EDU_EVAL_OUTPUT" \
    --docs "$FINEWEB_EDU_EVAL_DOCS"

mkdir -p "$FINEWEB_EDU_TASK_DIR"
cat > "$FINEWEB_EDU_TASK_DIR/${FINEWEB_EDU_TASK_NAME}.yaml" <<YAML
task: ${FINEWEB_EDU_TASK_NAME}
dataset_path: json
dataset_kwargs:
  data_files:
    validation: ${FINEWEB_EDU_EVAL_OUTPUT}
output_type: loglikelihood_rolling
validation_split: validation
doc_to_text: ""
doc_to_target: "text"
metric_list:
  - metric: word_perplexity
    aggregation: weighted_perplexity
    higher_is_better: false
  - metric: byte_perplexity
    aggregation: weighted_perplexity
    higher_is_better: false
  - metric: bits_per_byte
    aggregation: bits_per_byte
    higher_is_better: false
metadata:
  version: 1.0
YAML

SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:-"--transformer-impl local --attention-backend unfused --no-persist-layer-norm --no-use-tokenizer-model-from-checkpoint-args --tokenizer-type Llama2Tokenizer --tokenizer-model $LLAMA2_TOKENIZER_MODEL"}
RUN_NAME=${RUN_NAME:-$(basename "$CKPT")-fineweb-edu-lm-eval-smoke10}
JOB_NAME=${JOB_NAME:-lm-eval-fineweb-edu}
SBATCH_PARTITION=${SBATCH_PARTITION:-debug}
SBATCH_TIME=${SBATCH_TIME:-00:25:00}
SBATCH_NODES=${SBATCH_NODES:-1}
SBATCH_NTASKS_PER_NODE=${SBATCH_NTASKS_PER_NODE:-1}
SBATCH_GPUS_PER_NODE=${SBATCH_GPUS_PER_NODE:-1}
SBATCH_CPUS_PER_TASK=${SBATCH_CPUS_PER_TASK:-72}
SBATCH_MEM=${SBATCH_MEM:-230000}
READY_MAX_TRIES=${READY_MAX_TRIES:-36}
READY_SLEEP_SECONDS=${READY_SLEEP_SECONDS:-5}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-cler-eval}
LM_EVAL_MAX_LENGTH=${LM_EVAL_MAX_LENGTH:-4096}

EXPORT_ARGS=(
    ALL
    SERVER_IMPL=static
    CONDA_ENV_NAME=$CONDA_ENV_NAME
    CKPT=$CKPT
    RUN_NAME=$RUN_NAME
    TASKS=$FINEWEB_EDU_TASK_NAME
    LLAMA2_TOKENIZER_MODEL=$LLAMA2_TOKENIZER_MODEL
    SERVER_EXTRA_ARGS=$SERVER_EXTRA_ARGS
    LM_EVAL_INCLUDE_PATH=$FINEWEB_EDU_TASK_DIR
    LM_EVAL_MAX_LENGTH=$LM_EVAL_MAX_LENGTH
    READY_MAX_TRIES=$READY_MAX_TRIES
    READY_SLEEP_SECONDS=$READY_SLEEP_SECONDS
)

if [ "${FINEWEB_FULL:-0}" != "1" ]; then
    LIMIT=${LIMIT:-10}
    EXPORT_ARGS+=(LIMIT=$LIMIT)
fi

IFS=,
EXPORT_STRING="${EXPORT_ARGS[*]}"
unset IFS

sbatch --parsable \
    --partition="$SBATCH_PARTITION" \
    --time="$SBATCH_TIME" \
    --nodes="$SBATCH_NODES" \
    --ntasks-per-node="$SBATCH_NTASKS_PER_NODE" \
    --gpus-per-node="$SBATCH_GPUS_PER_NODE" \
    --cpus-per-task="$SBATCH_CPUS_PER_TASK" \
    --mem="$SBATCH_MEM" \
    --job-name="$JOB_NAME" \
    --export="$EXPORT_STRING" \
    _research/eval/run_lm_eval_local_api.sbatch
