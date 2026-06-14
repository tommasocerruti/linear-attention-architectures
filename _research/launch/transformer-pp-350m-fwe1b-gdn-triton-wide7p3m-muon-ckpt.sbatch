#!/bin/bash
# Capacity control: plain hybrid GDN (Triton) + ~7.3M extra params via FFN width (2816->2936),
# Muon. Matched to full-rank CLER-H (+7.3M, job 2437524, best 2.82203). If this matches full-rank CLER-H the
# gain is capacity; if full-rank CLER-H beats it the routed error contributes. Compare to GDN baseline 2418194
# (FFN 2816, best 2.82857).
#
#   sbatch _research/launch/transformer-pp-350m-fwe1b-gdn-triton-wide7p3m-muon-ckpt.sbatch
#
#SBATCH --job-name=gdn-triton-wide7p3m-350m
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=02:58:00
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
export WANDB_GROUP=350m-fwe1b-gdn-capacity-control-20260601
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=fineweb_edu_1b_llama2
export WANDB_ARCH=gdn-triton-wide-plus7p3m
export WANDB_KERNEL=triton

export APERTUS_FEATURE=gdn-triton-wide7p3m-350m-fwe1b-muon-ckpt
export APERTUS_TRACK=350m-fwe1b-gdn-triton-wide7p3m-muon
export LINEAR_VARIANT=gdn_triton
export EXP_NAME=gdn-triton-wide7p3m-350m-llama2-fwe1b-muon-ckpt

export TRAIN_SAMPLES=244140
export EVAL_INTERVAL=100
export EVAL_ITERS=10
export EXIT_DURATION_IN_MINS=170
export MBS=2
export GBS=128
export LINEAR_ATTENTION_FREQ=3
export FFN_HIDDEN=2936          # 2816 + 120 => ~+7.37M params over 20 layers (~ full-rank CLER-H +7.34M)

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
