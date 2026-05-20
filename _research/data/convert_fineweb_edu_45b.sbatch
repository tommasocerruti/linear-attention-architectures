#!/bin/bash
#
# Download a deterministic 45B-token FineWeb-Edu prefix and convert it to
# Megatron binary format with a LLaMA-2 SentencePiece tokenizer.
#
# Output: _research/results/data/fineweb_edu/fineweb_edu_45b_llama2/
#
#SBATCH --job-name=convert-fineweb-edu-45b
#SBATCH --account=lsaie-ss26
#SBATCH --partition=xfer
#SBATCH --time=24:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=240000
#SBATCH --no-requeue

export FINEWEB_EDU_TOKENS=45000000000
export FINEWEB_EDU_NAME=fineweb_edu_45b_llama2

exec bash "$(dirname "$0")/convert_fineweb_edu.sbatch"
