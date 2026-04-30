#!/bin/bash
#
# Quick 1-GPU Gated DeltaNetPytorch hybrid on GPT-2-tokenized ClimbMix.
# Intended for short engineering comparisons against the softmax baseline.
#
# Default dataset prefix:
#   /capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
#
# Override by exporting MEGATRON_DATA_PATH to a different Megatron-binary prefix
# without the .bin/.idx suffix.
#
#SBATCH --job-name=quick-gdn-1gpu-climbmix
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=115000
#SBATCH --no-requeue

set -euo pipefail

DEFAULT_MEGATRON_DATA_PATH=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
MEGATRON_DATA_PATH=${MEGATRON_DATA_PATH:-$DEFAULT_MEGATRON_DATA_PATH}
if [ ! -f "${MEGATRON_DATA_PATH}.idx" ] || [ ! -f "${MEGATRON_DATA_PATH}.bin" ]; then
    echo "ERROR: dataset prefix not found: $MEGATRON_DATA_PATH" >&2
    exit 1
fi

echo "START TIME: $(date)"
echo "MEGATRON_DATA_PATH: $MEGATRON_DATA_PATH"
export APERTUS_PHASE_SBATCH_START=$(date +%s.%N)

SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" -o | sed -n 's/.* Command=\([^ ]*\).*/\1/p')
if [ -n "$SCRIPT_PATH" ]; then
    REPO_DIR=$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)
else
    REPO_DIR=${CLER_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
fi
WORKDIR=$REPO_DIR
SCRATCH_ROOT=${SCRATCH:-/iopsstor/scratch/cscs/$USER}
SCRATCH_REPO_DIR=$SCRATCH_ROOT/cler
PACKAGE_DIR=$SCRATCH_REPO_DIR/_research/packages
DATASET_CACHE_DIR=$SCRATCH_REPO_DIR/cache/dataset
TMP_CACHE_DIR=$SCRATCH_REPO_DIR/cache/tmp

cd $WORKDIR

export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR${PYTHONPATH:+:$PYTHONPATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CPP_LOG_LEVEL=WARNING
export TRITON_CACHE_DIR=$SCRATCH_REPO_DIR/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH_REPO_DIR/cache/inductor
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=$TMP_CACHE_DIR

mkdir -p \
    $PACKAGE_DIR \
    $DATASET_CACHE_DIR \
    $TMP_CACHE_DIR \
    $TRITON_CACHE_DIR \
    $TORCHINDUCTOR_CACHE_DIR \
    $SCRATCH_REPO_DIR/_research/results/runs \
    $SCRATCH_REPO_DIR/_research/results/performance \
    $WORKDIR/_research/results/runs

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=25678

NUM_LAYERS=6
HIDDEN=512
FFN_HIDDEN=1408
NUM_HEADS=8
NUM_KV_HEADS=2

MBS=4
GBS=8
SEQ_LEN=1024
TRAIN_ITERS=200
EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-40}
LR=3e-4
MIN_LR=3e-5

TP=1
PP=1
CP=1

PROJECT=${WANDB_PROJECT:-megatron-lm-research-baseline}
GIT_SHORT_HASH=$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo nogit)
EXP_NAME=${EXP_NAME:-quick-gdn-1gpu-climbmix-$GIT_SHORT_HASH}

export APERTUS_LOG_DIR=$SCRATCH_REPO_DIR/_research/results/performance
export APERTUS_FEATURE=gated-deltanet-pytorch
export APERTUS_TRACK=${APERTUS_TRACK:-quick-climbmix-1gpu}
export APERTUS_RUN_NAME=$EXP_NAME-$SLURM_JOB_ID
export APERTUS_LOG_ACT_STATS=0
export APERTUS_LOG_TOP1_ACC=0

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --main-grads-dtype fp32
)

NETWORK_SIZE_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN
    --ffn-hidden-size $FFN_HIDDEN
    --num-attention-heads $NUM_HEADS
    --group-query-attention
    --num-query-groups $NUM_KV_HEADS
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope
    --rotary-base 500000
    --make-vocab-size-divisible-by 128
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --attention-backend auto --deterministic-mode
    --seq-length $SEQ_LEN
)

GDN_ARGS=(
    --experimental-attention-variant gated_delta_net_pytorch
    --linear-attention-freq 3
    --linear-conv-kernel-dim 4
    --linear-key-head-dim 64
    --linear-value-head-dim 64
    --linear-num-key-heads 4
    --linear-num-value-heads 8
    --apply-layernorm-1p
    --apply-wd-to-qk-layernorm
    --attention-output-gate
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-iters $TRAIN_ITERS
    --exit-duration-in-mins $EXIT_DURATION_IN_MINS
    --log-interval 1
    --eval-interval $TRAIN_ITERS
    --eval-iters 0
    --optimizer adam
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 50
    
    --disable-bias-linear
    --no-check-for-nan-in-loss-and-grad
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style cosine
    --lr-warmup-iters 20
    --lr-decay-iters $TRAIN_ITERS
)

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --context-parallel-size $CP
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
    --log-memory-to-tensorboard
    --log-params-norm
    --tensorboard-dir $SCRATCH_REPO_DIR/_research/results/runs/tb-$SLURM_JOB_ID
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
)

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/_research/data/gpt2-vocab.json
    --merge-file $WORKDIR/_research/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $MEGATRON_DATA_PATH
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 4
)

MEGATRON_ARGS="\
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${GDN_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_PROJECT=$PROJECT
    export WANDB_NAME=$EXP_NAME-$SLURM_JOB_ID
    MEGATRON_ARGS="$MEGATRON_ARGS --wandb-project $PROJECT --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID --wandb-save-dir $SCRATCH_REPO_DIR/_research/results/runs"
else
    export WANDB_MODE=disabled
fi

ENV_SETUP="export MASTER_ADDR=$MASTER_ADDR; \
    export MASTER_PORT=$MASTER_PORT; \
    export RANK=\$SLURM_PROCID; \
    export WORLD_SIZE=\$SLURM_NTASKS; \
    export LOCAL_RANK=\$SLURM_LOCALID; \
    export LOCAL_WORLD_SIZE=$((SLURM_NTASKS/SLURM_NNODES)); \
    export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR\${PYTHONPATH:+:\$PYTHONPATH}; \
    cd $WORKDIR"

TRAINING_CMD="bash $WORKDIR/_research/launch/install_python_deps.sh $PACKAGE_DIR && python3 $WORKDIR/pretrain_gpt.py $MEGATRON_ARGS"

echo "CMD: $TRAINING_CMD"
export APERTUS_PHASE_PRE_SRUN=$(date +%s.%N)
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=$WORKDIR/_research/launch/alps3.toml \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 120 \
    bash -c "export APERTUS_PHASE_CONTAINER_STARTED=\$(date +%s.%N); $ENV_SETUP; $TRAINING_CMD"

echo "END TIME: $(date)"
