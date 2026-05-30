#!/bin/bash
#
# Final pure all-GDN 350M / 1B-token CLER attempt.
# Triton/FLA GDN + CLER, dense_mean routing, per-head gamma, raw residual, Muon.
# Checkpoints are saved under _research/results/checkpoints.
#
# Submit after debug smoke passes:
#   sbatch _research/launch/transformer-pp-350m-fwe1b-pure-cler-gdn-triton-densemean-headgamma-muon-ckpt.sbatch
#
#SBATCH --job-name=pure-cler-350m
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=05:58:00
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
export WANDB_GROUP=350m-fwe1b-dense-cler-final-20260528
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=fineweb_edu_1b_llama2
export WANDB_ARCH=cler-gdn-triton-pure-densemean-headgamma
export WANDB_KERNEL=triton

export APERTUS_FEATURE=pure-cler-gdn-densemean-headgamma-350m-fwe1b-muon-triton-ckpt
export APERTUS_TRACK=350m-fwe1b-dense-cler-final-muon-triton
export LINEAR_VARIANT=cler_gdn_triton
export EXP_NAME=pure-cler-gdn-densemean-headgamma-350m-llama2-fwe1b-muon-triton-ckpt

export TRAIN_SAMPLES=244140
export EVAL_INTERVAL=100
export EVAL_ITERS=10
export EXIT_DURATION_IN_MINS=350
export MBS=2
export GBS=128
export LINEAR_ATTENTION_FREQ='[1]*20'

export CLER_GAMMA_INIT=0.01
export CLER_GAMMA_MODE=head
export CLER_ROUTING_MODE=dense_mean
export CLER_NORMALIZE_RESIDUAL=0
export CLER_RESIDUAL_NORM_EPS=1e-6
export CLER_GAMMA_LOG_INTERVAL=10
export CLER_RESIDUAL_LOG_INTERVAL=10

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
