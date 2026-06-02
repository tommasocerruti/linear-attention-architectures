#!/bin/bash
#SBATCH --job-name=probe-clerv-mag
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:15:00
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
export WANDB_MODE=offline
export MEGATRON_CLER_LOG_MAG=1
export CLER_REPO_ROOT=/users/course_00252/cler
export MEGATRON_DATA_PATH=/iopsstor/scratch/cscs/course_00252/cler/_research/results/data/fineweb_edu/fineweb_edu_1000000000_llama2/fineweb_edu_1000000000_llama2_text_document
export LLAMA2_TOKENIZER_MODEL=/iopsstor/scratch/cscs/course_00252/llama2-tokenizer/tokenizer.model
export WANDB_ENTITY=cler; export WANDB_PROJECT=clerv1-runs; export WANDB_GROUP=probe; export WANDB_ARCH=probe; export WANDB_KERNEL=triton; export WANDB_DATASET=fwe1b
export LINEAR_VARIANT=cler_gdn_triton
export CLER_HIDDEN_ROUTING=1
export CLER_HIDDEN_ROUTE_VALUE=1
export EXP_NAME=probe-clerv-mag
export TRAIN_SAMPLES=244140
export EVAL_INTERVAL=100000; export EVAL_ITERS=8
export EXIT_DURATION_IN_MINS=12
export MBS=2; export GBS=128; export LINEAR_ATTENTION_FREQ=3
export ENABLE_CHECKPOINTING=1
export LOAD_DIR=/iopsstor/scratch/cscs/course_00252/cler/_research/results/checkpoints/cler-gdn-hidden-routeval-full-350m-llama2-fwe1b-muon-ckpt-2453393
export SAVE_DIR=/iopsstor/scratch/cscs/course_00252/cler/_research/results/checkpoints/probe-clerv-mag-${SLURM_JOB_ID:-x}
export SAVE_INTERVAL=100000; export CKPT_FORMAT=torch
export MEGATRON_LINEAR_TORCH_COMPILE=0
cd /users/course_00252/cler
exec bash _research/launch/transformer-pp-350m-linear-muon.sbatch
