#!/bin/bash
#
# Transformer++ 350M baseline on 1 GH200 node (4 GPUs).
# 24L / 1024H / 2560F / 16h / 4kv (GQA).
# SwiGLU, RMSNorm, RoPE, NorMuon (adaptive_muon + normuon), WSD, bf16.
# 15B tokens on ClimbMix, ~28610 steps at 524K tokens/step.
# Muon recipe validated on 1B-token ablation: NorMuon beats AdamW by ~0.10 nats.
# --overlap-param-gather intentionally omitted (breaks Muon, see README).
#
#SBATCH --job-name=transformer-pp-350m-muon
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
: "${MEGATRON_DATA_PATH:?please set MEGATRON_DATA_PATH to the FineWeb-Edu Megatron-binary dataset prefix (without .bin/.idx), e.g. /path/to/fineweb_edu_15b_llama2_text_document}"
: "${LLAMA2_TOKENIZER_MODEL:?please set LLAMA2_TOKENIZER_MODEL to a local LLaMA-2 tokenizer.model path}"
if [ ! -f "$LLAMA2_TOKENIZER_MODEL" ]; then
    echo "ERROR: tokenizer model not found: $LLAMA2_TOKENIZER_MODEL" >&2
    exit 1
fi
echo "START TIME: $(date)"
export APERTUS_PHASE_SBATCH_START=$(date +%s.%N)

REPO_DIR=/iopsstor/scratch/cscs/$USER/cler
WORKDIR=${CLER_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$REPO_DIR}}
if [ ! -f "$WORKDIR/pretrain_gpt.py" ]; then
    WORKDIR=$REPO_DIR
fi
if [ ! -f "$WORKDIR/pretrain_gpt.py" ]; then
    echo "ERROR: CLER repo not found. Set CLER_REPO_ROOT to the checkout containing pretrain_gpt.py." >&2
    exit 1
fi
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

NUM_LAYERS=${NUM_LAYERS:-24}
HIDDEN=${HIDDEN:-1024}
FFN_HIDDEN=${FFN_HIDDEN:-2560}
NUM_HEADS=${NUM_HEADS:-16}
NUM_KV_HEADS=${NUM_KV_HEADS:-4}

MBS=${MBS:-16}
GBS=${GBS:-128}
SEQ_LEN=${SEQ_LEN:-4096}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-244141}
TRAIN_ITERS=${TRAIN_ITERS:-}
EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-595}
# MUON_LR drives both the main lr and scalar lr at the validated 3.6e-4 / 1.5e-3 ratio (~4.17x).
MUON_LR=${MUON_LR:-3.6e-4}
LR=$MUON_LR
MIN_LR=$(python3 -c "print(f'{float(\"$MUON_LR\")/10:.2e}')")
MUON_SCALAR_LR=$(python3 -c "print(f'{float(\"$MUON_LR\")*1.5e-3/3.6e-4:.2e}')")

TP=1
PP=1
CP=${CP:-1}

PROJECT=${WANDB_PROJECT:-clerv1-runs}
EXP_NAME=${EXP_NAME:-transformer-pp-350m-muon-$(git rev-parse --short HEAD)}

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
        --lr-warmup-samples 25600
        --lr-wsd-decay-samples 732422
    )
fi
EVAL_ITERS=${EVAL_ITERS:-0}

export APERTUS_LOG_DIR=$WORKDIR/_research/results/performance
export APERTUS_FEATURE=transformer-pp
export APERTUS_TRACK=transformer-pp-350m
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

# NorMuon recipe (validated on 1B-token 350M ablation).
MUON_ARGS=(
    --muon-scale-mode spectral
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

CHECKPOINT_ARGS=(
    --save $WORKDIR/_research/results/checkpoints/$EXP_NAME
    --save-interval ${SAVE_INTERVAL:-1000}
    --ckpt-format torch
)
mkdir -p $WORKDIR/_research/results/checkpoints/$EXP_NAME

MEGATRON_ARGS="\
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${MUON_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${RECOMPUTE_ARGS[@]+"${RECOMPUTE_ARGS[@]}"} \
    ${CHECKPOINT_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

export WANDB_ARCH=softmax
export WANDB_KERNEL=triton
export WANDB_OPTIMIZER=muon
export WANDB_MODEL_SIZE=350M
export WANDB_SEQ_LEN=$SEQ_LEN
export WANDB_IS_SMOKE=${WANDB_IS_SMOKE:-0}
export WANDB_ENTITY=${WANDB_ENTITY:-cler}

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_PROJECT=$PROJECT
    export WANDB_NAME=$EXP_NAME-$SLURM_JOB_ID
    MEGATRON_ARGS="$MEGATRON_ARGS --wandb-project $PROJECT --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID --wandb-save-dir $WORKDIR/_research/results/runs"
    if [ -n "${WANDB_ENTITY:-}" ]; then
        MEGATRON_ARGS="$MEGATRON_ARGS --wandb-entity $WANDB_ENTITY"
    fi
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
