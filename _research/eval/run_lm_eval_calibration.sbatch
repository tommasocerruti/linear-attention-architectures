#!/bin/bash
#
# Start the native Megatron local-completions server and run a small set of
# loglikelihood calibration probes against the same API contract used by lm_eval.
#
# Required exports:
#   CKPT=/path/to/megatron/checkpoint/root
#   LLAMA2_TOKENIZER_MODEL=/path/to/tokenizer.model
#
# Optional exports:
#   WORKDIR=/path/to/cler checkout used for the server
#   CALIBRATION_SCRIPT=/path/to/calibrate_lm_eval_api.py
#   SERVER_EXTRA_ARGS="..."
#
#SBATCH --job-name=lm-eval-calibration
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:25:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/eval/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/eval/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=230000
#SBATCH --no-requeue

set -euo pipefail

: "${CKPT:?set CKPT to a Megatron checkpoint root dir}"
: "${LLAMA2_TOKENIZER_MODEL:?set LLAMA2_TOKENIZER_MODEL to tokenizer.model}"

REPO_DIR=/iopsstor/scratch/cscs/$USER/cler
WORKDIR=${WORKDIR:-${SLURM_SUBMIT_DIR:-$REPO_DIR}}
cd "$WORKDIR"

SERVER_PY_TAG=$(python3 - <<'PY'
import sys
print("py%d%d" % (sys.version_info.major, sys.version_info.minor))
PY
)

PACKAGE_DIR=${SERVER_PACKAGE_DIR:-$REPO_DIR/_research/packages-server-$SERVER_PY_TAG}
SERVER_PYTHONPATH=$WORKDIR:$PACKAGE_DIR
TMP_CACHE_DIR=$REPO_DIR/cache/tmp
TRITON_CACHE_DIR=$REPO_DIR/cache/triton
TORCHINDUCTOR_CACHE_DIR=$REPO_DIR/cache/inductor
OUT_ROOT=$REPO_DIR/_research/results/eval

PORT=${PORT:-5000}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
READY_MAX_TRIES=${READY_MAX_TRIES:-36}
READY_SLEEP_SECONDS=${READY_SLEEP_SECONDS:-5}
SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:-}
CALIBRATION_SCRIPT=${CALIBRATION_SCRIPT:-$WORKDIR/_research/eval/calibrate_lm_eval_api.py}

SERVER_LOG=$OUT_ROOT/lm-eval-calibration-server-${SLURM_JOB_ID}.log
CALIBRATION_JSON=$OUT_ROOT/lm-eval-calibration-${SLURM_JOB_ID}.json

mkdir -p \
    "$PACKAGE_DIR" \
    "$TMP_CACHE_DIR" \
    "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$OUT_ROOT"

export PYTHONPATH="$SERVER_PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CPP_LOG_LEVEL=WARNING
export TRITON_CACHE_DIR
export TORCHINDUCTOR_CACHE_DIR
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$TMP_CACHE_DIR

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

python3 "$CALIBRATION_SCRIPT" --base-url "http://$SERVER_HOST:$PORT" | tee "$CALIBRATION_JSON"
echo "Calibration JSON: $CALIBRATION_JSON"
echo "Server log: $SERVER_LOG"
