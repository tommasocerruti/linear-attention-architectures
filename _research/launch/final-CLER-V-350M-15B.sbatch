#!/bin/bash
# CLER-V-350M-15B-MUON: 350M FineWeb-Edu(_tc) Muon run. Auto-generated final-experiment launcher (2026-06-10).
#SBATCH --job-name=CLER-V-350M-15B-MUON
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail
source /users/lingfeng/.wandb_cler_env

export CLER_REPO_ROOT=/users/lingfeng/cler
export MEGATRON_DATA_PATH=/iopsstor/scratch/cscs/lingfeng/cler/_research/results/data/fineweb_edu/fineweb_edu_62b_llama2_tc/fineweb_edu_62b_llama2_tc_text_document
export LLAMA2_TOKENIZER_MODEL=/iopsstor/scratch/cscs/lingfeng/llama2-tokenizer/tokenizer.model

export WANDB_ENTITY=cler
export WANDB_PROJECT=clerv1-runs
export WANDB_GROUP=350M-15B-FINAL-20260610
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=fineweb_edu_llama2_tc
export WANDB_ARCH=cler-v-full
export WANDB_KERNEL=triton
export APERTUS_FEATURE=CLER-V-350M-15B-MUON
export APERTUS_TRACK=CLER-V-350M-15B-MUON
export EXP_NAME=CLER-V-350M-15B-MUON

export LINEAR_VARIANT=cler_gdn_triton
export CLER_HIDDEN_ROUTING=1
export CLER_HIDDEN_ROUTE_VALUE=1

export TRAIN_SAMPLES=3662109
export EVAL_INTERVAL=500
export EVAL_ITERS=10
export EXIT_DURATION_IN_MINS=660
export MBS=2
export GBS=128
export LINEAR_ATTENTION_FREQ=3

export ENABLE_CHECKPOINTING=1
export SAVE_INTERVAL=500
export SAVE_RETAIN_INTERVAL=29000
export CKPT_FORMAT=torch
export SAVE_DIR=${SAVE_DIR:-/iopsstor/scratch/cscs/lingfeng/cler/_research/results/checkpoints/$EXP_NAME}
export LOAD_DIR=${LOAD_DIR:-$SAVE_DIR}

export MEGATRON_DISABLE_JIT_FUSER=1
export MEGATRON_INIT_PROCESS_GROUP_DEVICE_ID=1
export MEGATRON_LINEAR_TORCH_COMPILE=1
export MEGATRON_LINEAR_TORCH_COMPILE_MODE=reduce-overhead
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=0

cd /users/lingfeng/cler
exec bash _research/launch/transformer-pp-350m-linear-muon.sbatch
