#!/bin/bash
#
# Run lm-eval-harness against a Megatron checkpoint by launching the local
# text generation server and querying it through the OpenAI-compatible
# /v1/completions endpoint.
#
# Required exports:
#   CKPT=/path/to/megatron/checkpoint/root
#   LLAMA2_TOKENIZER_MODEL=/path/to/tokenizer.model
#
# Optional exports:
#   RUN_NAME=<results directory name>
#   TASKS=hellaswag,piqa,winogrande
#   PORT=5000
#   COORD_PORT=5001
#   SERVER_HOST=127.0.0.1
#   LIMIT=10
#   CONDA_ENV_NAME=<host-side conda env for lm_eval>
#   CONDA_SH=~/miniconda3/etc/profile.d/conda.sh
#   SERVER_IMPL=dynamic|static
#   SERVER_EXTRA_ARGS="..."
#   READY_MAX_TRIES=36
#   READY_SLEEP_SECONDS=5
#   LM_EVAL_REMOTE_TOKENIZER_DELAY_SECONDS=0.005
#   LM_EVAL_REMOTE_TOKENIZER_CACHE_SIZE=50000
#
#SBATCH --job-name=lm-eval-local-api
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/eval/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/eval/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail

: "${CKPT:?set CKPT to a Megatron checkpoint root dir}"
: "${LLAMA2_TOKENIZER_MODEL:?set LLAMA2_TOKENIZER_MODEL to tokenizer.model}"

REPO_DIR=/iopsstor/scratch/cscs/$USER/cler
WORKDIR=${SLURM_SUBMIT_DIR:-$REPO_DIR}
cd "$WORKDIR"

SERVER_PY_TAG=$(python3 - <<'PY'
import sys
print(f"py{sys.version_info.major}{sys.version_info.minor}")
PY
)

PACKAGE_DIR=${SERVER_PACKAGE_DIR:-$REPO_DIR/_research/packages-server-$SERVER_PY_TAG}
EVAL_PACKAGE_DIR=$REPO_DIR/_research/packages-eval
EVAL_SITE_DIR=$REPO_DIR/_research/eval/pythonpath
SERVER_PYTHONPATH=$WORKDIR:$PACKAGE_DIR
TMP_CACHE_DIR=$REPO_DIR/cache/tmp
TRITON_CACHE_DIR=$REPO_DIR/cache/triton
TORCHINDUCTOR_CACHE_DIR=$REPO_DIR/cache/inductor
OUT_ROOT=$REPO_DIR/_research/results/eval

RUN_NAME=${RUN_NAME:-$(basename "$CKPT")}
TASKS=${TASKS:-hellaswag,piqa,winogrande}
PORT=${PORT:-5000}
COORD_PORT=${COORD_PORT:-5001}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
LIMIT=${LIMIT:-}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-}
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
SERVER_IMPL=${SERVER_IMPL:-dynamic}
SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:-}
READY_MAX_TRIES=${READY_MAX_TRIES:-120}
READY_SLEEP_SECONDS=${READY_SLEEP_SECONDS:-5}
LM_EVAL_REMOTE_TOKENIZER_DELAY_SECONDS=${LM_EVAL_REMOTE_TOKENIZER_DELAY_SECONDS:-0.005}
LM_EVAL_REMOTE_TOKENIZER_CACHE_SIZE=${LM_EVAL_REMOTE_TOKENIZER_CACHE_SIZE:-50000}

SERVER_LOG=$OUT_ROOT/${RUN_NAME}-server-${SLURM_JOB_ID}.log
OUTDIR=$OUT_ROOT/$RUN_NAME

mkdir -p \
    "$PACKAGE_DIR" \
    "$EVAL_PACKAGE_DIR" \
    "$EVAL_SITE_DIR" \
    "$TMP_CACHE_DIR" \
    "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$OUTDIR"

export PYTHONPATH="$SERVER_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CPP_LOG_LEVEL=WARNING
export TRITON_CACHE_DIR
export TORCHINDUCTOR_CACHE_DIR
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$TMP_CACHE_DIR

run_in_eval_env() {
    if [ -n "$CONDA_ENV_NAME" ]; then
        (
            unset PYTHONPATH
            source "$CONDA_SH"
            conda activate "$CONDA_ENV_NAME"
            export PYTHONPATH="$EVAL_SITE_DIR"
            export LM_EVAL_REMOTE_TOKENIZER_DELAY_SECONDS
            export LM_EVAL_REMOTE_TOKENIZER_CACHE_SIZE
            "$@"
        )
    else
        (
            export PYTHONPATH="$EVAL_SITE_DIR:$EVAL_PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}"
            export LM_EVAL_REMOTE_TOKENIZER_DELAY_SECONDS
            export LM_EVAL_REMOTE_TOKENIZER_CACHE_SIZE
            "$@"
        )
    fi
}

if [ -n "$CONDA_ENV_NAME" ]; then
    if ! run_in_eval_env python -c "import lm_eval" >/dev/null 2>&1; then
        run_in_eval_env python -m pip install --quiet 'lm_eval[api]'
    fi
else
    if command -v uv >/dev/null 2>&1; then
        INSTALL="uv pip install --quiet"
    else
        INSTALL="python3 -m pip install --quiet"
    fi

    if ! PYTHONPATH="$EVAL_PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 -c "import lm_eval" >/dev/null 2>&1; then
        $INSTALL --target="$EVAL_PACKAGE_DIR" 'lm_eval[api]'
    fi
fi

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-25679}

ENV_SETUP="export MASTER_ADDR=$MASTER_ADDR; \
    export MASTER_PORT=$MASTER_PORT; \
    export RANK=\$SLURM_PROCID; \
    export WORLD_SIZE=\$SLURM_NTASKS; \
    export LOCAL_RANK=\$SLURM_LOCALID; \
    export LOCAL_WORLD_SIZE=$((SLURM_NTASKS/SLURM_NNODES)); \
    export PYTHONPATH=$SERVER_PYTHONPATH; \
    cd $WORKDIR"

if [ "$SERVER_IMPL" = "dynamic" ]; then
    SERVER_CMD="bash $WORKDIR/_research/launch/install_python_deps.sh $PACKAGE_DIR && \
        if ! PYTHONPATH=$SERVER_PYTHONPATH python3 -c 'import quart' >/dev/null 2>&1; then \
            python3 -m pip install --quiet --target=$PACKAGE_DIR quart; \
        fi && \
        python3 $WORKDIR/tools/run_dynamic_text_generation_server.py \
            --use-checkpoint-args \
            --load $CKPT \
            --model-provider gpt \
            --bf16 \
            --no-load-optim \
            --no-load-rng \
            --port $PORT \
            --host $SERVER_HOST \
            --inference-coordinator-port $COORD_PORT \
            --inference-max-seq-length 4096 \
            --inference-dynamic-batching-max-requests 64 \
            $SERVER_EXTRA_ARGS"
else
    SERVER_CMD="bash $WORKDIR/_research/launch/install_python_deps.sh $PACKAGE_DIR && \
        if ! PYTHONPATH=$SERVER_PYTHONPATH python3 -c 'import flask, flask_restful, tqdm' >/dev/null 2>&1; then \
            python3 -m pip install --quiet --target=$PACKAGE_DIR flask flask-restful tqdm; \
        fi && \
        python3 $WORKDIR/tools/run_text_generation_server.py \
            --use-checkpoint-args \
            --load $CKPT \
            --bf16 \
            --no-load-optim \
            --no-load-rng \
            --port $PORT \
            --host $SERVER_HOST \
            --inference-max-requests 64 \
            --inference-max-seq-length 4096 \
            $SERVER_EXTRA_ARGS"
fi

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" >/dev/null 2>&1 || true
        wait "$SERVER_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

srun -lu --mpi=pmix --network=disable_rdzv_get --environment=$WORKDIR/_research/launch/alps3.toml \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 120 \
    bash -c "$ENV_SETUP; $SERVER_CMD" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

READY=0
for _ in $(seq 1 "$READY_MAX_TRIES"); do
    if curl -sf -X POST "http://$SERVER_HOST:$PORT/v1/completions" \
        -H 'Content-Type: application/json' \
        -d '{"prompt":"Hello","max_tokens":1}' >/dev/null; then
        READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        break
    fi
    sleep "$READY_SLEEP_SECONDS"
done

if [ "$READY" != "1" ]; then
    echo "Server did not become ready."
    tail -n 200 "$SERVER_LOG" || true
    exit 1
fi

if [ -n "$CONDA_ENV_NAME" ]; then
    LM_EVAL_ARGS=(
        python -m lm_eval
        --model local-completions
        --model_args "model=cler,base_url=http://$SERVER_HOST:$PORT/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False"
        --batch_size 8
        --tasks "$TASKS"
        --output_path "$OUTDIR"
        --log_samples
    )
else
    LM_EVAL_ARGS=(
        python3 -m lm_eval
        --model local-completions
        --model_args "model=cler,base_url=http://$SERVER_HOST:$PORT/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False"
        --batch_size 8
        --tasks "$TASKS"
        --output_path "$OUTDIR"
        --log_samples
    )
fi

if [ -n "$LIMIT" ]; then
    LM_EVAL_ARGS+=(--limit "$LIMIT")
fi

run_in_eval_env "${LM_EVAL_ARGS[@]}"
