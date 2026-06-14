#!/bin/bash
#SBATCH --job-name=probe-proj-norms
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:12:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
set -euo pipefail
cd /users/course_00252/cler
PKG=/iopsstor/scratch/cscs/course_00252/cler/_research/packages
srun -lu --mpi=pmix --network=disable_rdzv_get \
     --environment=/users/course_00252/cler/_research/launch/alps3.toml \
     bash -lc "bash /users/course_00252/cler/_research/launch/install_python_deps.sh $PKG >/dev/null 2>&1; \
               export PYTHONPATH=/users/course_00252/cler:$PKG; \
               python3 /users/course_00252/cler/_research/launch/probe_proj_norms.py"
