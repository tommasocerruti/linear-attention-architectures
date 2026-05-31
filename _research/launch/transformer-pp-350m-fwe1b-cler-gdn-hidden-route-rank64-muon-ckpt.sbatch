#!/bin/bash
# 350M / 1B-token HYBRID GDN + CLER-H, LOW-RANK projection rank=64, Muon.
# Second point in the CLER-H rank sweep: full (7.3M, job 2437524, best 2.8220) / r=16 (+344K, job
# 2441838) / r=64 (+1.38M, this run). Shows how the projection rank affects the gain.
# Compare to hybrid GDN baseline 2418194 (best 2.82857).
#
#   sbatch _research/launch/transformer-pp-350m-fwe1b-cler-gdn-hidden-route-rank64-muon-ckpt.sbatch
#
#SBATCH --job-name=cler-hidden-rank64-350m
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=03:30:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail

source /users/course_00252/.wandb_cler_env
if [ -f /users/course_00252/.hf_cler_env ]; then
    source /users/course_00252/.hf_cler_env
fi

export CLER_REPO_ROOT=/users/course_00252/cler
export MEGATRON_DATA_PATH=/iopsstor/scratch/cscs/course_00252/cler/_research/results/data/fineweb_edu/fineweb_edu_1000000000_llama2/fineweb_edu_1000000000_llama2_text_document
export LLAMA2_TOKENIZER_MODEL=/iopsstor/scratch/cscs/course_00252/llama2-tokenizer/tokenizer.model

export WANDB_ENTITY=cler
export WANDB_PROJECT=clerv1-runs
export WANDB_GROUP=350m-fwe1b-cler-hidden-route-20260531
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=fineweb_edu_1b_llama2
export WANDB_ARCH=cler-gdn-hidden-route-rank64
export WANDB_KERNEL=triton

export APERTUS_FEATURE=cler-gdn-hidden-route-rank64-350m-fwe1b-muon-ckpt
export APERTUS_TRACK=350m-fwe1b-cler-hidden-route-rank64-muon
export LINEAR_VARIANT=cler_gdn_triton
export CLER_HIDDEN_ROUTING=1
export CLER_HIDDEN_RANK=64
export EXP_NAME=cler-gdn-hidden-route-rank64-350m-llama2-fwe1b-muon-ckpt

export TRAIN_SAMPLES=244140
export EVAL_INTERVAL=100
export EVAL_ITERS=10
export EXIT_DURATION_IN_MINS=200
export MBS=2
export GBS=128
export LINEAR_ATTENTION_FREQ=3

export ENABLE_CHECKPOINTING=1
export SAVE_INTERVAL=500
export CKPT_FORMAT=torch
export NO_SAVE_OPTIM=0
export NO_SAVE_RNG=0
export SAVE_DIR=/iopsstor/scratch/cscs/course_00252/cler/_research/results/checkpoints/$EXP_NAME-${SLURM_JOB_ID:-manual}

export MEGATRON_LINEAR_TORCH_COMPILE=1
export MEGATRON_LINEAR_TORCH_COMPILE_MODE=reduce-overhead
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=0

cd /users/course_00252/cler
exec bash _research/launch/transformer-pp-350m-linear-muon.sbatch
