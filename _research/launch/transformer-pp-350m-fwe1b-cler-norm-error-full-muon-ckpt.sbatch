#!/bin/bash
# CLER-H ERROR-vs-REPRESENTATION CONTROL: route the raw value v (full-rank Linear(512->1024)
# per GDN layer x14 = +7,340,032 params = CLER-H full budget EXACTLY) instead of the write residual r.
# Tests error-vs-representation: if v helps like r it is not the error specifically; if v is flat it is.
# CLER-H r=64 -- rank=256 gives EXACTLY +7,340,032 params (= CLER-H full-rank, 14 GDN layers * 524288).
# If this matches CLER-H full-rank (job 2437524, best 2.82203) the gain is capacity; if CLER-H beats it

#
#   sbatch _research/launch/transformer-pp-350m-fwe1b-cler-gdn-hidden-norm-error-full-muon-ckpt.sbatch
#
#SBATCH --job-name=cler-norm-error-full-350m
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
export WANDB_ARCH=cler-gdn-hidden-norm-error-full
export WANDB_KERNEL=triton

export APERTUS_FEATURE=cler-gdn-hidden-norm-error-full-350m-fwe1b-muon-ckpt
export APERTUS_TRACK=350m-fwe1b-cler-norm-error-full-muon
export LINEAR_VARIANT=cler_gdn_triton
export CLER_HIDDEN_ROUTING=1
export CLER_HIDDEN_ROUTE_VALUE=0
export CLER_HIDDEN_NORMALIZE_INPUT=1
export EXP_NAME=cler-gdn-hidden-norm-error-full-350m-llama2-fwe1b-muon-ckpt

export TRAIN_SAMPLES=244140
export EVAL_INTERVAL=100
export EVAL_ITERS=10
export EXIT_DURATION_IN_MINS=170
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
