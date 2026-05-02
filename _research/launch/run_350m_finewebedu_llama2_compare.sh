#!/bin/bash
#
# Submit-time wrapper for the 350M FineWeb-Edu/LLaMA2 comparison.
# It keeps the sbatch command short because Clariden's submit filter can
# reject long --export lists with a misleading "invalid partition" error.

set -euo pipefail

variant=${1:?usage: run_350m_finewebedu_llama2_compare.sh softmax|gdn|cler|softmax_muon|gdn_muon|cler_muon}

source /users/course_00252/.wandb_cler_env
if [ -f /users/course_00252/.hf_cler_env ]; then
    source /users/course_00252/.hf_cler_env
fi

export CLER_REPO_ROOT=${CLER_REPO_ROOT:-/users/course_00252/cler}
export MEGATRON_DATA_PATH=${MEGATRON_DATA_PATH:-/iopsstor/scratch/cscs/course_00252/cler/_research/results/data/fineweb_edu/fineweb_edu_1000000000_llama2/fineweb_edu_1000000000_llama2_text_document}
export LLAMA2_TOKENIZER_MODEL=${LLAMA2_TOKENIZER_MODEL:-/iopsstor/scratch/cscs/course_00252/llama2-tokenizer/tokenizer.model}

export WANDB_ENTITY=${WANDB_ENTITY:-cler}
export WANDB_PROJECT=${WANDB_PROJECT:-clerv1-runs}
export WANDB_GROUP=${WANDB_GROUP:-350m-llama2-finewebedu-1b-6way-20260502}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DATASET=${WANDB_DATASET:-fineweb_edu_1b_llama2}
export WANDB_IS_SMOKE=${WANDB_IS_SMOKE:-0}

export TRAIN_SAMPLES=${TRAIN_SAMPLES:-244140}
export EVAL_INTERVAL=${EVAL_INTERVAL:-100}
export EVAL_ITERS=${EVAL_ITERS:-10}
export EXIT_DURATION_IN_MINS=${EXIT_DURATION_IN_MINS:-170}
export MEGATRON_LINEAR_TORCH_COMPILE=${MEGATRON_LINEAR_TORCH_COMPILE:-1}
export MEGATRON_LINEAR_TORCH_COMPILE_MODE=${MEGATRON_LINEAR_TORCH_COMPILE_MODE:-reduce-overhead}

cd "$CLER_REPO_ROOT"

case "$variant" in
    softmax)
        export EXP_NAME=${EXP_NAME:-350m-llama2-fwe1b-softmax-adamw}
        exec bash _research/launch/transformer-pp-350m-adamw.sbatch
        ;;
    gdn)
        export EXP_NAME=${EXP_NAME:-350m-llama2-fwe1b-gdn-pytorch-adamw}
        exec bash _research/launch/transformer-pp-350m-gdn-pytorch.sbatch
        ;;
    cler)
        export EXP_NAME=${EXP_NAME:-350m-llama2-fwe1b-cler-v1-adamw}
        exec bash _research/launch/transformer-pp-350m-cler-v1.sbatch
        ;;
    softmax_muon)
        export EXP_NAME=${EXP_NAME:-350m-llama2-fwe1b-softmax-muon}
        exec bash _research/launch/transformer-pp-350m-muon.sbatch
        ;;
    gdn_muon)
        export EXP_NAME=${EXP_NAME:-350m-llama2-fwe1b-gdn-pytorch-muon}
        exec bash _research/launch/transformer-pp-350m-gdn-pytorch-muon.sbatch
        ;;
    cler_muon)
        export EXP_NAME=${EXP_NAME:-350m-llama2-fwe1b-cler-v1-muon}
        exec bash _research/launch/transformer-pp-350m-cler-v1-muon.sbatch
        ;;
    *)
        echo "unknown variant: $variant" >&2
        exit 2
        ;;
esac
