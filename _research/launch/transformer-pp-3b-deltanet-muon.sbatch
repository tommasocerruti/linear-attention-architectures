#!/bin/bash
#
# Hybrid DeltaNet ~3B baseline on 8 GH200 nodes (32 GPUs).
# 32L / 3072H / 8192F / 24h / 8kv (GQA for SDPA layers), with a 2:1 DeltaNet:SDPA layer pattern.
# DeltaNet layers use 8 q/k heads x 64 dims and 8 value heads x 128 dims.
# SwiGLU, RMSNorm, RoPE, NorMuon (adaptive_muon + normuon), WSD, bf16.
# 60B tokens, 32 nodes (128 GPUs), TP=1, DP=128, GBS=2048, MBS=4, GAS=4.
#
#SBATCH --job-name=transformer-pp-3b-deltanet-muon
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail
: "${MEGATRON_DATA_PATH:?please set MEGATRON_DATA_PATH to the Megatron-binary dataset prefix (without .bin/.idx), e.g. /path/to/climbmix_small}"
: "${LLAMA2_TOKENIZER_MODEL:?please set LLAMA2_TOKENIZER_MODEL to a local LLaMA-2 tokenizer.model path}"
if [ ! -f "$LLAMA2_TOKENIZER_MODEL" ]; then
    echo "ERROR: tokenizer model not found: $LLAMA2_TOKENIZER_MODEL" >&2
    exit 1
fi
echo "START TIME: $(date)"
export APERTUS_PHASE_SBATCH_START=$(date +%s.%N)

REPO_DIR=/iopsstor/scratch/cscs/$USER/cler
WORKDIR=${SLURM_SUBMIT_DIR:-$REPO_DIR}
cd $WORKDIR

PACKAGE_DIR=$REPO_DIR/_research/packages
DATASET_CACHE_DIR=$REPO_DIR/cache/dataset
TMP_CACHE_DIR=$REPO_DIR/cache/tmp

export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR${PYTHONPATH:+:$PYTHONPATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CPP_LOG_LEVEL=WARNING
export TRITON_CACHE_DIR=$REPO_DIR/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$REPO_DIR/cache/inductor
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=$TMP_CACHE_DIR

mkdir -p $DATASET_CACHE_DIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TMP_CACHE_DIR $PACKAGE_DIR _research/results/runs _research/results/performance

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=25678

NUM_LAYERS=${NUM_LAYERS:-32}
HIDDEN=${HIDDEN:-3072}
FFN_HIDDEN=${FFN_HIDDEN:-8192}
NUM_HEADS=${NUM_HEADS:-24}
NUM_KV_HEADS=${NUM_KV_HEADS:-8}

MBS=${MBS:-4}
GBS=${GBS:-2048}
SEQ_LEN=${SEQ_LEN:-4096}
# 60B tokens / 4096 tokens-per-sample = 14,648,437 samples
TRAIN_SAMPLES=${TRAIN_SAMPLES:-14648437}
TRAIN_ITERS=${TRAIN_ITERS:-}
EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-1430}
LR=${LR:-5e-4}
MIN_LR=${MIN_LR:-5e-5}

TP=1
PP=1
CP=${CP:-1}

PROJECT=${WANDB_PROJECT:-megatron-lm-research-baseline}
EXP_NAME=${EXP_NAME:-transformer-pp-3b-deltanet-muon-$(git rev-parse --short HEAD)}

if [ -n "$TRAIN_ITERS" ]; then
    TRAIN_STOP_ARGS=(--train-iters $TRAIN_ITERS)
    _default_eval=$(( TRAIN_ITERS < 2000 ? TRAIN_ITERS : 2000 ))
    EVAL_INTERVAL=${EVAL_INTERVAL:-$_default_eval}
    LR_DURATION_ARGS=(
        --lr-warmup-iters ${LR_WARMUP_ITERS:-200}
        --lr-decay-iters ${LR_DECAY_ITERS:-$TRAIN_ITERS}
        --lr-wsd-decay-iters ${LR_WSD_DECAY_ITERS:-4000}
    )
else
    TRAIN_STOP_ARGS=(--train-samples $TRAIN_SAMPLES)
    # eval every 2000 iters worth of samples = 2000 * GBS
    EVAL_INTERVAL=${EVAL_INTERVAL:-$(( 2000 * GBS ))}
    # 1% warmup, 10% WSD decay of 60B tokens
    LR_DURATION_ARGS=(
        --lr-warmup-samples ${LR_WARMUP_SAMPLES:-145408}
        --lr-wsd-decay-samples ${LR_WSD_DECAY_SAMPLES:-1464320}
    )
fi
EVAL_ITERS=${EVAL_ITERS:-0}

SAVE_DIR=${SAVE_DIR:-$REPO_DIR/_research/results/checkpoints/$EXP_NAME}
# LOAD_DIR must be set explicitly to resume from a checkpoint; no implicit resume.
if [ -n "$TRAIN_ITERS" ] && [ "$TRAIN_ITERS" -lt 2000 ]; then
    SAVE_INTERVAL=${SAVE_INTERVAL:-$TRAIN_ITERS}
else
    SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
fi
CKPT_FORMAT=${CKPT_FORMAT:-}
NO_SAVE_OPTIM=${NO_SAVE_OPTIM:-0}
NO_SAVE_RNG=${NO_SAVE_RNG:-0}
CHECKPOINT_ARGS=()
if [ -n "$SAVE_DIR" ]; then
    mkdir -p "$SAVE_DIR"
    CHECKPOINT_ARGS=(
        --save "$SAVE_DIR"
        --save-interval "$SAVE_INTERVAL"
    )
    if [ -n "${LOAD_DIR:-}" ] && [ -d "${LOAD_DIR}" ] && [ -n "$(ls -A "$LOAD_DIR" 2>/dev/null)" ]; then
        CHECKPOINT_ARGS+=(--load "$LOAD_DIR")
    fi
fi
if [ -n "$CKPT_FORMAT" ]; then
    CHECKPOINT_ARGS+=(--ckpt-format "$CKPT_FORMAT")
fi
if [ "$NO_SAVE_OPTIM" = "1" ]; then
    CHECKPOINT_ARGS+=(--no-save-optim)
fi
if [ "$NO_SAVE_RNG" = "1" ]; then
    CHECKPOINT_ARGS+=(--no-save-rng)
fi

export APERTUS_LOG_DIR=$WORKDIR/_research/results/performance
export APERTUS_FEATURE=delta-net
export APERTUS_TRACK=${APERTUS_TRACK:-transformer-pp-3b-deltanet-muon}
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
    --attention-backend auto
    --seq-length $SEQ_LEN
)

DELTA_NET_ARGS=(
    --experimental-attention-variant delta_net
    --linear-attention-freq ${LINEAR_ATTENTION_FREQ:-3}
    --linear-conv-kernel-dim ${LINEAR_CONV_KERNEL_DIM:-4}
    --linear-key-head-dim ${LINEAR_KEY_HEAD_DIM:-64}
    --linear-value-head-dim ${LINEAR_VALUE_HEAD_DIM:-128}
    --linear-num-key-heads ${LINEAR_NUM_KEY_HEADS:-8}
    --linear-num-value-heads ${LINEAR_NUM_VALUE_HEADS:-8}
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    ${TRAIN_STOP_ARGS[@]}
    --exit-duration-in-mins $EXIT_DURATION_IN_MINS
    --log-interval 1
    --eval-interval $EVAL_INTERVAL
    --eval-iters $EVAL_ITERS
    --optimizer adaptive_muon
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 50
    --cross-entropy-loss-fusion
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

MUON_ARGS=(
    --muon-scale-mode spectral
    --muon-nesterov
    --muon-momentum 0.95
    --muon-scalar-lr 1.5e-3
    --muon-scalar-weight-decay 0.0
    --adaptive-muon-moment2-method normuon
)

LEARNING_RATE_ARGS=(
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style WSD
    --lr-wsd-decay-style minus_sqrt
    ${LR_DURATION_ARGS[@]}
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
    --use-distributed-optimizer
    --overlap-grad-reduce
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
    --log-memory-to-tensorboard
    --log-params-norm
    --tensorboard-dir $WORKDIR/_research/results/runs/tb-$SLURM_JOB_ID
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
)

TOKENIZER_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model "$LLAMA2_TOKENIZER_MODEL"
)

DATA_ARGS=(
    --data-path $MEGATRON_DATA_PATH
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 4
)

RECOMPUTE_ARGS=()
if [ "${RECOMPUTE:-0}" = "1" ]; then
    RECOMPUTE_ARGS=(
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers $NUM_LAYERS
    )
fi

MEGATRON_ARGS="\
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${DELTA_NET_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${MUON_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${RECOMPUTE_ARGS[@]+"${RECOMPUTE_ARGS[@]}"} \
    ${CHECKPOINT_ARGS[@]+"${CHECKPOINT_ARGS[@]}"} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

export WANDB_ARCH=deltanet
export WANDB_KERNEL=triton
export WANDB_OPTIMIZER=muon
export WANDB_MODEL_SIZE=3B
export WANDB_SEQ_LEN=$SEQ_LEN
export WANDB_IS_SMOKE=${WANDB_IS_SMOKE:-0}

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_PROJECT=$PROJECT
    export WANDB_NAME=$EXP_NAME-$SLURM_JOB_ID
    MEGATRON_ARGS="$MEGATRON_ARGS --wandb-project $PROJECT --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID --wandb-save-dir $WORKDIR/_research/results/runs"
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
