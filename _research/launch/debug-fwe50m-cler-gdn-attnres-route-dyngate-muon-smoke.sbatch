#!/bin/bash
# Debug smoke: hybrid GDN + CLER AttnRes-style routing with a DATA-DEPENDENT (dynamic-gate) self-bias
# (cler_routing_mode=attnres + cler_dynamic_gate=1), Muon. The convex softmax attention over
# {value} + prior GDN write residuals uses a per-token Linear(value) self-bias instead of a static one.
#
#   sbatch _research/launch/debug-fwe50m-cler-gdn-attnres-route-dyngate-muon-smoke.sbatch
#
#SBATCH --job-name=debug-cler-attnres-dyngate
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
export WANDB_GROUP=debug-cler-attnres-dyngate-20260530
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=fineweb_edu_50m_llama2_smoke
export WANDB_ARCH=cler-gdn-attnres-route-dyngate
export WANDB_KERNEL=triton
export WANDB_IS_SMOKE=1

export APERTUS_FEATURE=cler-gdn-attnres-route-dyngate-muon-smoke
export APERTUS_TRACK=debug-350m-cler-gdn-attnres-route-dyngate-muon
export LINEAR_VARIANT=cler_gdn_triton
export CLER_ROUTING_MODE=attnres
export CLER_DYNAMIC_GATE=1
export EXP_NAME=cler-gdn-attnres-route-dyngate-350m-fwe50m-muon-smoke

export TRAIN_ITERS=10
export EVAL_INTERVAL=5
export EVAL_ITERS=1
export EXIT_DURATION_IN_MINS=25
export MBS=2
export GBS=128
export LINEAR_ATTENTION_FREQ=3

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
