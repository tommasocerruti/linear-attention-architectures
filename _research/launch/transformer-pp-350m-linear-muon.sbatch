#!/bin/bash
#
# Shared 350M Linear-RNN Muon launcher for FineWeb-Edu/LLaMA-2 runs.
# Select the architecture with LINEAR_VARIANT:
#   gdn_pytorch  - PyTorch Gated DeltaNet baseline, CLER off
#   gdn_triton   - Triton/FLA Gated DeltaNet baseline, CLER off
#   cler         - CLER v1 on PyTorch Gated DeltaNet
#   cler_gdn_triton - CLER v1 on Triton/FLA Gated DeltaNet
#   deltanet     - DeltaNet baseline
#   deltanet_pytorch - pure PyTorch DeltaNet baseline
#   cler_deltanet_pytorch - CLER v1 on pure PyTorch DeltaNet
#   cler_deltanet_triton - CLER v1 on Triton/FLA DeltaNet
#   linear_transformer_pytorch - pure additive linear-transformer baseline
#
# Required environment:
#   MEGATRON_DATA_PATH=/path/to/fineweb_edu_*_text_document
#   LLAMA2_TOKENIZER_MODEL=/path/to/tokenizer.model
#
# Muon recipe follows the validated NorMuon launcher:
#   adaptive_muon + normuon, spectral scaling, nesterov, scalar LR 1.5e-3,
#   no QKV split, no overlap-param-gather.
#
#SBATCH --job-name=transformer-pp-350m-linear-muon
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
: "${MEGATRON_DATA_PATH:?please set MEGATRON_DATA_PATH to the FineWeb-Edu Megatron-binary dataset prefix (without .bin/.idx)}"
: "${LLAMA2_TOKENIZER_MODEL:?please set LLAMA2_TOKENIZER_MODEL to a local LLaMA-2 tokenizer.model path}"
if [ ! -f "$LLAMA2_TOKENIZER_MODEL" ]; then
    echo "ERROR: tokenizer model not found: $LLAMA2_TOKENIZER_MODEL" >&2
    exit 1
fi
if [ ! -f "${MEGATRON_DATA_PATH}.idx" ] || [ ! -f "${MEGATRON_DATA_PATH}.bin" ]; then
    echo "ERROR: dataset prefix not found: $MEGATRON_DATA_PATH" >&2
    exit 1
fi

echo "START TIME: $(date)"
export APERTUS_PHASE_SBATCH_START=$(date +%s.%N)

REPO_ROOT=${CLER_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/iopsstor/scratch/cscs/$USER/cler}}
if [ ! -f "$REPO_ROOT/pretrain_gpt.py" ]; then
    REPO_ROOT=/iopsstor/scratch/cscs/$USER/cler
fi
if [ ! -f "$REPO_ROOT/pretrain_gpt.py" ]; then
    echo "ERROR: CLER repo not found. Set CLER_REPO_ROOT to the checkout containing pretrain_gpt.py." >&2
    exit 1
fi
WORKDIR=$REPO_ROOT
SCRATCH_REPO_DIR=/iopsstor/scratch/cscs/$USER/cler
PACKAGE_DIR=${PACKAGE_DIR:-$SCRATCH_REPO_DIR/_research/packages}
DATASET_CACHE_DIR=$SCRATCH_REPO_DIR/cache/dataset
TMP_CACHE_DIR=$SCRATCH_REPO_DIR/cache/tmp

cd "$WORKDIR"

export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR${PYTHONPATH:+:$PYTHONPATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CPP_LOG_LEVEL=WARNING
export TRITON_CACHE_DIR=$SCRATCH_REPO_DIR/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH_REPO_DIR/cache/inductor
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
export MEGATRON_LINEAR_TORCH_COMPILE=${MEGATRON_LINEAR_TORCH_COMPILE:-1}
export MEGATRON_LINEAR_TORCH_COMPILE_MODE=${MEGATRON_LINEAR_TORCH_COMPILE_MODE:-reduce-overhead}
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=${MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS:-0}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=$TMP_CACHE_DIR

mkdir -p \
    "$PACKAGE_DIR" \
    "$DATASET_CACHE_DIR" \
    "$TMP_CACHE_DIR" \
    "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$SCRATCH_REPO_DIR/_research/results/runs" \
    "$SCRATCH_REPO_DIR/_research/results/performance" \
    "$WORKDIR/_research/results/runs"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=25678

LINEAR_VARIANT=${LINEAR_VARIANT:-gdn_pytorch}

NUM_LAYERS=${NUM_LAYERS:-20}
HIDDEN=${HIDDEN:-1024}
FFN_HIDDEN=${FFN_HIDDEN:-2816}
NUM_HEADS=${NUM_HEADS:-16}
NUM_KV_HEADS=${NUM_KV_HEADS:-4}

MBS=${MBS:-2}
GBS=${GBS:-128}
SEQ_LEN=${SEQ_LEN:-4096}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-3662109}
TRAIN_ITERS=${TRAIN_ITERS:-}
EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-595}

MUON_LR=${MUON_LR:-3.6e-4}
LR=${LR:-$MUON_LR}
MIN_LR=${MIN_LR:-3.6e-5}
MUON_SCALAR_LR=${MUON_SCALAR_LR:-1.5e-3}

TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-1}

PROJECT=${WANDB_PROJECT:-clerv1-runs}
GIT_SHA=$(git -C "$WORKDIR" rev-parse --short HEAD 2>/dev/null || echo unknown)

LINEAR_ARGS=()
CLER_ARGS=()
# Full Attention Residuals (AttnRes) passthrough, independent of the mixer variant.
ATTN_RES_ARGS=()
if [ "${ATTN_RES_ENABLED:-0}" = "1" ]; then
    ATTN_RES_ARGS=(--attn-res-enabled)
fi
WANDB_ARCH_DEFAULT=
APERTUS_FEATURE_DEFAULT=
APERTUS_TRACK_DEFAULT=
EXP_NAME_DEFAULT=

case "$LINEAR_VARIANT" in
    gdn_triton)
        export WANDB_KERNEL=${WANDB_KERNEL:-triton}
        LINEAR_ARGS=(
            --experimental-attention-variant gated_delta_net
            --linear-attention-freq ${LINEAR_ATTENTION_FREQ:-3}
            --linear-conv-kernel-dim ${LINEAR_CONV_KERNEL_DIM:-4}
            --linear-key-head-dim ${LINEAR_KEY_HEAD_DIM:-64}
            --linear-value-head-dim ${LINEAR_VALUE_HEAD_DIM:-64}
            --linear-num-key-heads ${LINEAR_NUM_KEY_HEADS:-4}
            --linear-num-value-heads ${LINEAR_NUM_VALUE_HEADS:-8}
            --attention-output-gate
        )
        WANDB_ARCH_DEFAULT=gdn-triton
        APERTUS_FEATURE_DEFAULT=gdn-triton-muon
        APERTUS_TRACK_DEFAULT=transformer-pp-350m-gdn-triton-muon
        EXP_NAME_DEFAULT=transformer-pp-350m-gdn-triton-muon-$GIT_SHA
        ;;
    gdn_pytorch)
        LINEAR_ARGS=(
            --experimental-attention-variant gated_delta_net_pytorch
            --linear-attention-freq ${LINEAR_ATTENTION_FREQ:-3}
            --linear-conv-kernel-dim ${LINEAR_CONV_KERNEL_DIM:-4}
            --linear-key-head-dim ${LINEAR_KEY_HEAD_DIM:-64}
            --linear-value-head-dim ${LINEAR_VALUE_HEAD_DIM:-64}
            --linear-num-key-heads ${LINEAR_NUM_KEY_HEADS:-4}
            --linear-num-value-heads ${LINEAR_NUM_VALUE_HEADS:-8}
            --attention-output-gate
        )
        WANDB_ARCH_DEFAULT=gdn-pytorch
        APERTUS_FEATURE_DEFAULT=gdn-pytorch-muon
        APERTUS_TRACK_DEFAULT=transformer-pp-350m-gdn-pytorch-muon
        EXP_NAME_DEFAULT=transformer-pp-350m-gdn-pytorch-muon-$GIT_SHA
        ;;
    cler)
        export CLER_ENABLED=1
        LINEAR_ARGS=(
            --experimental-attention-variant gated_delta_net_pytorch
            --linear-attention-freq ${LINEAR_ATTENTION_FREQ:-3}
            --linear-conv-kernel-dim ${LINEAR_CONV_KERNEL_DIM:-4}
            --linear-key-head-dim ${LINEAR_KEY_HEAD_DIM:-64}
            --linear-value-head-dim ${LINEAR_VALUE_HEAD_DIM:-64}
            --linear-num-key-heads ${LINEAR_NUM_KEY_HEADS:-4}
            --linear-num-value-heads ${LINEAR_NUM_VALUE_HEADS:-8}
            --attention-output-gate
        )
        CLER_ARGS=(
            --cler-enabled
            --cler-gamma-init ${CLER_GAMMA_INIT:-0.0}
            --cler-gamma-mode ${CLER_GAMMA_MODE:-scalar}
            --cler-routing-mode ${CLER_ROUTING_MODE:-latest}
            --cler-residual-norm-eps ${CLER_RESIDUAL_NORM_EPS:-1e-6}
        )
        if [ "${CLER_DETACH_RESIDUAL:-0}" = "1" ]; then
            CLER_ARGS+=(--cler-detach-residual)
        fi
        if [ "${CLER_NORMALIZE_RESIDUAL:-0}" = "1" ]; then
            CLER_ARGS+=(--cler-normalize-residual)
        fi
        WANDB_ARCH_DEFAULT=cler-v1-gdn-pytorch
        APERTUS_FEATURE_DEFAULT=cler-v1-gdn-pytorch-muon
        APERTUS_TRACK_DEFAULT=transformer-pp-350m-cler-v1-muon
        EXP_NAME_DEFAULT=transformer-pp-350m-cler-v1-muon-$GIT_SHA
        ;;
    cler_gdn_triton)
        export CLER_ENABLED=1
        export WANDB_KERNEL=${WANDB_KERNEL:-triton}
        LINEAR_ARGS=(
            --experimental-attention-variant gated_delta_net
            --linear-attention-freq ${LINEAR_ATTENTION_FREQ:-3}
            --linear-conv-kernel-dim ${LINEAR_CONV_KERNEL_DIM:-4}
            --linear-key-head-dim ${LINEAR_KEY_HEAD_DIM:-64}
            --linear-value-head-dim ${LINEAR_VALUE_HEAD_DIM:-64}
            --linear-num-key-heads ${LINEAR_NUM_KEY_HEADS:-4}
            --linear-num-value-heads ${LINEAR_NUM_VALUE_HEADS:-8}
            --attention-output-gate
        )
        CLER_ARGS=(
            --cler-enabled
            --cler-gamma-init ${CLER_GAMMA_INIT:-0.0}
            --cler-gamma-mode ${CLER_GAMMA_MODE:-scalar}
            --cler-routing-mode ${CLER_ROUTING_MODE:-latest}
            --cler-residual-norm-eps ${CLER_RESIDUAL_NORM_EPS:-1e-6}
        )
        if [ "${CLER_DETACH_RESIDUAL:-0}" = "1" ]; then
            CLER_ARGS+=(--cler-detach-residual)
        fi
        if [ "${CLER_NORMALIZE_RESIDUAL:-0}" = "1" ]; then
            CLER_ARGS+=(--cler-normalize-residual)
        fi
        WANDB_ARCH_DEFAULT=cler-v1-gdn-triton
        APERTUS_FEATURE_DEFAULT=cler-v1-gdn-triton-muon
        APERTUS_TRACK_DEFAULT=transformer-pp-350m-cler-gdn-triton-muon
        EXP_NAME_DEFAULT=transformer-pp-350m-cler-gdn-triton-muon-$GIT_SHA
        ;;
    deltanet)
        LINEAR_ARGS=(
            --experimental-attention-variant delta_net
            --linear-attention-freq ${LINEAR_ATTENTION_FREQ:-3}
            --linear-conv-kernel-dim ${LINEAR_CONV_KERNEL_DIM:-4}
            --linear-key-head-dim ${LINEAR_KEY_HEAD_DIM:-64}
            --linear-value-head-dim ${LINEAR_VALUE_HEAD_DIM:-128}
            --linear-num-key-heads ${LINEAR_NUM_KEY_HEADS:-4}
            --linear-num-value-heads ${LINEAR_NUM_VALUE_HEADS:-4}
        )
        WANDB_ARCH_DEFAULT=dn
        APERTUS_FEATURE_DEFAULT=delta-net-muon
        APERTUS_TRACK_DEFAULT=transformer-pp-350m-deltanet-muon
        EXP_NAME_DEFAULT=transformer-pp-350m-deltanet-muon-$GIT_SHA
        ;;
    deltanet_pytorch)
        LINEAR_ARGS=(
            --experimental-attention-variant delta_net_pytorch
            --linear-attention-freq ${LINEAR_ATTENTION_FREQ:-3}
            --linear-conv-kernel-dim ${LINEAR_CONV_KERNEL_DIM:-4}
            --linear-key-head-dim ${LINEAR_KEY_HEAD_DIM:-64}
            --linear-value-head-dim ${LINEAR_VALUE_HEAD_DIM:-128}
            --linear-num-key-heads ${LINEAR_NUM_KEY_HEADS:-4}
            --linear-num-value-heads ${LINEAR_NUM_VALUE_HEADS:-4}
        )
        WANDB_ARCH_DEFAULT=dn-pytorch
        APERTUS_FEATURE_DEFAULT=delta-net-pytorch-muon
        APERTUS_TRACK_DEFAULT=transformer-pp-350m-deltanet-pytorch-muon
        EXP_NAME_DEFAULT=transformer-pp-350m-deltanet-pytorch-muon-$GIT_SHA
        ;;
    cler_deltanet_pytorch)
        export CLER_ENABLED=1
        LINEAR_ARGS=(
            --experimental-attention-variant cler_delta_net_pytorch
            --linear-attention-freq ${LINEAR_ATTENTION_FREQ:-3}
            --linear-conv-kernel-dim ${LINEAR_CONV_KERNEL_DIM:-4}
            --linear-key-head-dim ${LINEAR_KEY_HEAD_DIM:-64}
            --linear-value-head-dim ${LINEAR_VALUE_HEAD_DIM:-128}
            --linear-num-key-heads ${LINEAR_NUM_KEY_HEADS:-4}
            --linear-num-value-heads ${LINEAR_NUM_VALUE_HEADS:-4}
        )
        CLER_ARGS=(
            --cler-enabled
            --cler-gamma-init ${CLER_GAMMA_INIT:-0.0}
            --cler-gamma-mode ${CLER_GAMMA_MODE:-scalar}
            --cler-routing-mode ${CLER_ROUTING_MODE:-latest}
            --cler-residual-norm-eps ${CLER_RESIDUAL_NORM_EPS:-1e-6}
        )
        if [ "${CLER_DETACH_RESIDUAL:-0}" = "1" ]; then
            CLER_ARGS+=(--cler-detach-residual)
        fi
        if [ "${CLER_NORMALIZE_RESIDUAL:-0}" = "1" ]; then
            CLER_ARGS+=(--cler-normalize-residual)
        fi
        WANDB_ARCH_DEFAULT=cler-v1-deltanet-pytorch
        APERTUS_FEATURE_DEFAULT=cler-v1-deltanet-pytorch-muon
        APERTUS_TRACK_DEFAULT=transformer-pp-350m-cler-deltanet-pytorch-muon
        EXP_NAME_DEFAULT=transformer-pp-350m-cler-deltanet-pytorch-muon-$GIT_SHA
        ;;
    cler_deltanet_triton)
        export CLER_ENABLED=1
        export WANDB_KERNEL=${WANDB_KERNEL:-triton}
        LINEAR_ARGS=(
            --experimental-attention-variant delta_net
            --linear-attention-freq ${LINEAR_ATTENTION_FREQ:-3}
            --linear-conv-kernel-dim ${LINEAR_CONV_KERNEL_DIM:-4}
            --linear-key-head-dim ${LINEAR_KEY_HEAD_DIM:-64}
            --linear-value-head-dim ${LINEAR_VALUE_HEAD_DIM:-128}
            --linear-num-key-heads ${LINEAR_NUM_KEY_HEADS:-4}
            --linear-num-value-heads ${LINEAR_NUM_VALUE_HEADS:-4}
        )
        CLER_ARGS=(
            --cler-enabled
            --cler-gamma-init ${CLER_GAMMA_INIT:-0.0}
            --cler-gamma-mode ${CLER_GAMMA_MODE:-scalar}
            --cler-routing-mode ${CLER_ROUTING_MODE:-latest}
            --cler-residual-norm-eps ${CLER_RESIDUAL_NORM_EPS:-1e-6}
        )
        if [ "${CLER_DETACH_RESIDUAL:-0}" = "1" ]; then
            CLER_ARGS+=(--cler-detach-residual)
        fi
        if [ "${CLER_NORMALIZE_RESIDUAL:-0}" = "1" ]; then
            CLER_ARGS+=(--cler-normalize-residual)
        fi
        WANDB_ARCH_DEFAULT=cler-v1-deltanet-triton
        APERTUS_FEATURE_DEFAULT=cler-v1-deltanet-triton-muon
        APERTUS_TRACK_DEFAULT=transformer-pp-350m-cler-deltanet-triton-muon
        EXP_NAME_DEFAULT=transformer-pp-350m-cler-deltanet-triton-muon-$GIT_SHA
        ;;
    linear_transformer_pytorch)
        LINEAR_ARGS=(
            --experimental-attention-variant linear_transformer_pytorch
            --linear-attention-freq ${LINEAR_ATTENTION_FREQ:-3}
            --linear-conv-kernel-dim ${LINEAR_CONV_KERNEL_DIM:-4}
            --linear-key-head-dim ${LINEAR_KEY_HEAD_DIM:-64}
            --linear-value-head-dim ${LINEAR_VALUE_HEAD_DIM:-128}
            --linear-num-key-heads ${LINEAR_NUM_KEY_HEADS:-4}
            --linear-num-value-heads ${LINEAR_NUM_VALUE_HEADS:-4}
        )
        WANDB_ARCH_DEFAULT=linear-transformer-pytorch
        APERTUS_FEATURE_DEFAULT=linear-transformer-pytorch-muon
        APERTUS_TRACK_DEFAULT=transformer-pp-350m-linear-transformer-pytorch-muon
        EXP_NAME_DEFAULT=transformer-pp-350m-linear-transformer-pytorch-muon-$GIT_SHA
        ;;
    *)
        echo "ERROR: unknown LINEAR_VARIANT='$LINEAR_VARIANT' (expected gdn_triton, gdn_pytorch, cler, cler_gdn_triton, deltanet, deltanet_pytorch, cler_deltanet_pytorch, cler_deltanet_triton, or linear_transformer_pytorch)" >&2
        exit 1
        ;;
esac

# Optional data-dependent CLER receiver gate (replaces static gamma), any CLER variant.
if [ "${CLER_DYNAMIC_GATE:-0}" = "1" ] && [ "${CLER_ENABLED:-0}" = "1" ]; then
    CLER_ARGS+=(--cler-dynamic-gate)
fi

# CLER-H: route the write residual into the shared hidden stream (not the value target).
if [ "${CLER_HIDDEN_ROUTING:-0}" = "1" ] && [ "${CLER_ENABLED:-0}" = "1" ]; then
    CLER_ARGS+=(--cler-hidden-routing)
    if [ -n "${CLER_HIDDEN_RANK:-}" ]; then
        CLER_ARGS+=(--cler-hidden-rank ${CLER_HIDDEN_RANK})
    fi
    # Capacity control: route a self-transform of the hidden state (no cross-layer error content).
    if [ "${CLER_HIDDEN_SELF_TRANSFORM:-0}" = "1" ]; then
        CLER_ARGS+=(--cler-hidden-self-transform)
    fi
    # Control: route the raw value v instead of the write residual (error vs representation).
    if [ "${CLER_HIDDEN_ROUTE_VALUE:-0}" = "1" ]; then
        CLER_ARGS+=(--cler-hidden-route-value)
    fi
fi

EXP_NAME=${EXP_NAME:-$EXP_NAME_DEFAULT}

CHECKPOINT_ARGS=()
if [ "${ENABLE_CHECKPOINTING:-0}" = "1" ]; then
    SAVE_DIR=${SAVE_DIR:-$SCRATCH_REPO_DIR/_research/results/checkpoints/$EXP_NAME-$SLURM_JOB_ID}
    SAVE_INTERVAL=${SAVE_INTERVAL:-500}
    CKPT_FORMAT=${CKPT_FORMAT:-torch}
    NO_SAVE_OPTIM=${NO_SAVE_OPTIM:-0}
    NO_SAVE_RNG=${NO_SAVE_RNG:-0}
    mkdir -p "$SAVE_DIR"
    CHECKPOINT_ARGS=(
        --save "$SAVE_DIR"
        --save-interval "$SAVE_INTERVAL"
    )
    if [ -n "${LOAD_DIR:-}" ]; then
        CHECKPOINT_ARGS+=(--load "$LOAD_DIR")
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
fi

if [ -n "$TRAIN_ITERS" ]; then
    TRAIN_STOP_ARGS=(--train-iters $TRAIN_ITERS)
    EVAL_INTERVAL=${EVAL_INTERVAL:-$TRAIN_ITERS}
    LR_DURATION_ARGS=(
        --lr-warmup-iters ${LR_WARMUP_ITERS:-5}
        --lr-decay-iters ${LR_DECAY_ITERS:-$TRAIN_ITERS}
        --lr-wsd-decay-iters ${LR_WSD_DECAY_ITERS:-10}
    )
else
    TRAIN_STOP_ARGS=(--train-samples $TRAIN_SAMPLES)
    EVAL_INTERVAL=${EVAL_INTERVAL:-$TRAIN_SAMPLES}
    LR_DURATION_ARGS=(
        --lr-warmup-samples ${LR_WARMUP_SAMPLES:-25600}
        --lr-wsd-decay-samples ${LR_WSD_DECAY_SAMPLES:-732422}
    )
fi
EVAL_ITERS=${EVAL_ITERS:-0}

export APERTUS_LOG_DIR=$SCRATCH_REPO_DIR/_research/results/performance
export APERTUS_FEATURE=${APERTUS_FEATURE:-$APERTUS_FEATURE_DEFAULT}
export APERTUS_TRACK=${APERTUS_TRACK:-$APERTUS_TRACK_DEFAULT}
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
    --muon-no-split-qkv
    --muon-nesterov
    --muon-momentum 0.95
    --muon-scalar-lr $MUON_SCALAR_LR
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
    --log-validation-ppl-to-tensorboard
    --tensorboard-dir $SCRATCH_REPO_DIR/_research/results/runs/tb-$SLURM_JOB_ID
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
    ${LINEAR_ARGS[@]} \
    ${CLER_ARGS[@]+"${CLER_ARGS[@]}"} \
    ${ATTN_RES_ARGS[@]+"${ATTN_RES_ARGS[@]}"} \
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

export WANDB_ENTITY=${WANDB_ENTITY:-cler}
export WANDB_PROJECT=$PROJECT
export WANDB_GROUP=${WANDB_GROUP:-clerv1}
export WANDB_ARCH=${WANDB_ARCH:-$WANDB_ARCH_DEFAULT}
export WANDB_KERNEL=${WANDB_KERNEL:-torch}
export WANDB_OPTIMIZER=muon
export WANDB_MODEL_SIZE=${WANDB_MODEL_SIZE:-350M}
export WANDB_SEQ_LEN=$SEQ_LEN
export WANDB_IS_SMOKE=${WANDB_IS_SMOKE:-0}
export WANDB_CLER_ENABLED=${CLER_ENABLED:-0}
export WANDB_CLER_GAMMA_INIT=${CLER_GAMMA_INIT:-0.0}
export WANDB_CLER_GAMMA_MODE=${CLER_GAMMA_MODE:-scalar}
export WANDB_CLER_ROUTING_MODE=${CLER_ROUTING_MODE:-latest}
export WANDB_CLER_NORMALIZE_RESIDUAL=${CLER_NORMALIZE_RESIDUAL:-0}
export WANDB_CLER_RESIDUAL_NORM_EPS=${CLER_RESIDUAL_NORM_EPS:-1e-6}
export WANDB_CLER_DETACH_RESIDUAL=${CLER_DETACH_RESIDUAL:-0}
export WANDB_CLER_DYNAMIC_GATE=${CLER_DYNAMIC_GATE:-0}
export WANDB_CLER_HIDDEN_ROUTING=${CLER_HIDDEN_ROUTING:-0}
export WANDB_ATTN_RES_ENABLED=${ATTN_RES_ENABLED:-0}
export WANDB_LINEAR_TORCH_COMPILE=${MEGATRON_LINEAR_TORCH_COMPILE:-0}

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_NAME=$EXP_NAME-$SLURM_JOB_ID
    MEGATRON_ARGS="$MEGATRON_ARGS --wandb-project $PROJECT --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID --wandb-save-dir $SCRATCH_REPO_DIR/_research/results/runs --wandb-entity $WANDB_ENTITY"
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
    export MEGATRON_LINEAR_TORCH_COMPILE=$MEGATRON_LINEAR_TORCH_COMPILE; \
    export MEGATRON_LINEAR_TORCH_COMPILE_MODE=$MEGATRON_LINEAR_TORCH_COMPILE_MODE; \
    export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=$MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS; \
    cd $WORKDIR"

TRAINING_CMD="bash $WORKDIR/_research/launch/install_python_deps.sh $PACKAGE_DIR && python3 $WORKDIR/pretrain_gpt.py $MEGATRON_ARGS"

echo "LINEAR_VARIANT: $LINEAR_VARIANT"
echo "MEGATRON_DATA_PATH: $MEGATRON_DATA_PATH"
echo "LLAMA2_TOKENIZER_MODEL: $LLAMA2_TOKENIZER_MODEL"
echo "CMD: $TRAINING_CMD"
export APERTUS_PHASE_PRE_SRUN=$(date +%s.%N)
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=$WORKDIR/_research/launch/alps3.toml \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 120 \
    bash -c "export APERTUS_PHASE_CONTAINER_STARTED=\$(date +%s.%N); $ENV_SETUP; $TRAINING_CMD"

echo "END TIME: $(date)"
