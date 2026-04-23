#!/bin/bash
#
# Short FineWeb-Edu conversion smoke for the LLaMA-2 tokenizer path.
# Reuses the full conversion script with a small deterministic token budget.
#
#SBATCH --job-name=convert-fineweb-edu-smoke
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

set -euo pipefail

REPO_ROOT=/iopsstor/scratch/cscs/$USER/cler
SCRIPT_DIR=$REPO_ROOT/_research/data
cd "$REPO_ROOT"

export FINEWEB_EDU_TOKENS=${FINEWEB_EDU_TOKENS:-50000000}
export FINEWEB_EDU_NAME=${FINEWEB_EDU_NAME:-fineweb_edu_50m_llama2_smoke}
export FINEWEB_EDU_WORKERS=${FINEWEB_EDU_WORKERS:-16}

exec bash "$SCRIPT_DIR/convert_fineweb_edu.sbatch"
