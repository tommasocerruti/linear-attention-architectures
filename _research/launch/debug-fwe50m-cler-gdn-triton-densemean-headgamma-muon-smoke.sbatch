#!/bin/bash
#
# Debug smoke for the final dense-routing CLER attempt.
# Triton/FLA GDN + CLER, dense_mean routing, per-head gamma, raw residual.
#
# Submit:
#   sbatch _research/launch/debug-fwe50m-cler-gdn-triton-densemean-headgamma-muon-smoke.sbatch
#
#SBATCH --job-name=debug-cler-dense
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:30:00
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
export MEGATRON_DATA_PATH=/iopsstor/scratch/cscs/course_00252/cler/_research/results/data/fineweb_edu/fineweb_edu_50m_llama2_smoke/fineweb_edu_50m_llama2_smoke_text_document
export LLAMA2_TOKENIZER_MODEL=/iopsstor/scratch/cscs/course_00252/llama2-tokenizer/tokenizer.model

export WANDB_ENTITY=cler
export WANDB_PROJECT=clerv1-runs
export WANDB_GROUP=debug-dense-cler-final-20260528
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=fineweb_edu_50m_llama2_smoke
export WANDB_ARCH=cler-gdn-triton-densemean-headgamma
export WANDB_KERNEL=triton
export WANDB_IS_SMOKE=1

export APERTUS_FEATURE=cler-gdn-densemean-headgamma-muon-triton-smoke
export APERTUS_TRACK=debug-350m-cler-gdn-densemean-headgamma-muon-triton
export LINEAR_VARIANT=cler_gdn_triton
export EXP_NAME=cler-gdn-densemean-headgamma-350m-fwe50m-muon-triton-smoke

export TRAIN_ITERS=10
export EVAL_INTERVAL=5
export EVAL_ITERS=1
export EXIT_DURATION_IN_MINS=25
export MBS=2
export GBS=128
export LINEAR_ATTENTION_FREQ=3

export CLER_GAMMA_INIT=0.01
export CLER_GAMMA_MODE=head
export CLER_ROUTING_MODE=dense_mean
export CLER_NORMALIZE_RESIDUAL=0
export CLER_RESIDUAL_NORM_EPS=1e-6
export CLER_GAMMA_LOG_INTERVAL=1
export CLER_RESIDUAL_LOG_INTERVAL=1

export ENABLE_CHECKPOINTING=1
export SAVE_INTERVAL=10
export CKPT_FORMAT=torch
export NO_SAVE_OPTIM=0
export NO_SAVE_RNG=0
export SAVE_DIR=/iopsstor/scratch/cscs/course_00252/cler/_research/results/checkpoints/$EXP_NAME-${SLURM_JOB_ID:-manual}

export MEGATRON_LINEAR_TORCH_COMPILE=1
export MEGATRON_LINEAR_TORCH_COMPILE_MODE=reduce-overhead
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=0

cd /users/course_00252/cler
exec bash _research/launch/transformer-pp-350m-linear-muon.sbatch
