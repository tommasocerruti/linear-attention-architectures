#!/bin/bash
#
# Minimal smoke test for the FLA dependency layout expected by GatedDeltaNet.
#
#SBATCH --job-name=fla-import-smoke
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/cler/_research/results/runs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --no-requeue

set -euo pipefail

WORKDIR=${SLURM_SUBMIT_DIR:-$PWD}
SCRATCH_ROOT=${SCRATCH:-/iopsstor/scratch/cscs/$USER}
SCRATCH_REPO_DIR=$SCRATCH_ROOT/cler
TMPPKG=/tmp/fla-smoke-pkgs

mkdir -p $SCRATCH_REPO_DIR/_research/results/runs
cd $WORKDIR

export PYTHONPATH=$WORKDIR${PYTHONPATH:+:$PYTHONPATH}

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=25678

ENV_SETUP="export MASTER_ADDR=$MASTER_ADDR; \
    export MASTER_PORT=$MASTER_PORT; \
    export RANK=\$SLURM_PROCID; \
    export WORLD_SIZE=\$SLURM_NTASKS; \
    export LOCAL_RANK=\$SLURM_LOCALID; \
    export LOCAL_WORLD_SIZE=$((SLURM_NTASKS/SLURM_NNODES)); \
    cd $WORKDIR"

SMOKE_CMD="rm -rf $TMPPKG && \
    bash $WORKDIR/_research/launch/install_python_deps.sh $TMPPKG && \
    export PYTHONPATH=$TMPPKG:$WORKDIR\${PYTHONPATH:+:\$PYTHONPATH} && \
    python3 -c \"import fla; from fla.modules.convolution import causal_conv1d; from fla.modules.l2norm import l2norm; from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print('SMOKE_OK', fla.__file__, causal_conv1d, l2norm, chunk_gated_delta_rule)\""

echo "CMD: $SMOKE_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=$WORKDIR/_research/launch/alps3.toml \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 120 \
    bash -c "$ENV_SETUP; $SMOKE_CMD"
