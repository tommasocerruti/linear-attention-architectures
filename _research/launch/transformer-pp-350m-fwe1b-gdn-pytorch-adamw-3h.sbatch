#!/bin/bash
#
# 350M PyTorch Gated DeltaNet AdamW on FineWeb-Edu 1B / LLaMA2 tokenizer.
#
# Queue with:
#   sbatch _research/launch/transformer-pp-350m-fwe1b-gdn-pytorch-adamw-3h.sbatch
#
# W&B: cler/clerv1-runs, group 350m-llama2-finewebedu-1b-6way-20260502
#
#SBATCH --job-name=350m-fwe1b-gdn-adamw
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail

cd /users/course_00252/cler
export EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-165}
exec _research/launch/run_350m_finewebedu_llama2_compare.sh gdn
