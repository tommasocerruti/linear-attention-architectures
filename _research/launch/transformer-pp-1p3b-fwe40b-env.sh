#!/bin/bash
#
# Shared 1.3B / 40B-token config for the AttnRes + CLER scaling runs (Muon, Triton/FLA GDN).
# Sourced by the per-variant transformer-pp-1p3b-fwe40b-*.sbatch wrappers AFTER they set the variant env
# (LINEAR_VARIANT / CLER_* / ATTN_RES_* / EXP_NAME / WANDB_ARCH / EXIT_DURATION_IN_MINS).
#
# =============================== EDIT THESE (per-user paths / account / W&B) ===============================
# 1) Your release checkout:
export CLER_REPO_ROOT=${CLER_REPO_ROOT:-/iopsstor/scratch/cscs/$USER/linear-attention-architectures}
# 2) Your 40-45B FineWeb-Edu (LLaMA2-tokenized) Megatron prefix. We could NOT see your scratch
#    from our account, so this is a placeholder. A ~45B FineWeb-Edu download script exists in the
#    repo (commit "added 45b fineweb_edu download"). The path must end at the *_text_document prefix.
export MEGATRON_DATA_PATH=${MEGATRON_DATA_PATH:-/iopsstor/scratch/cscs/$USER/REPLACE_ME_fineweb_edu_40b_llama2/REPLACE_ME_text_document}
# 3) Your LLaMA2 tokenizer.model:
export LLAMA2_TOKENIZER_MODEL=${LLAMA2_TOKENIZER_MODEL:-/iopsstor/scratch/cscs/$USER/llama2-tokenizer/tokenizer.model}
# 4) Your W&B (a DIFFERENT project than ours). Source your wandb env (with WANDB_API_KEY) or export it.
[ -f "/users/$USER/.wandb_cler_env" ] && source "/users/$USER/.wandb_cler_env"
export WANDB_ENTITY=${WANDB_ENTITY:-REPLACE_ME_wandb_entity}
export WANDB_PROJECT=${WANDB_PROJECT:-REPLACE_ME_wandb_project}
export WANDB_MODE=${WANDB_MODE:-online}
# ===============================================================================

export WANDB_GROUP=${WANDB_GROUP:-1p3b-fwe40b-attnres-cler-20260530}
export WANDB_DATASET=${WANDB_DATASET:-fineweb_edu_40b_llama2}
export WANDB_KERNEL=triton

# ---- 1.3B model: 24L / 2048H / 5632FFN (matches the repo's existing 1.3B launchers) ----
export NUM_LAYERS=24
export HIDDEN=2048
export FFN_HIDDEN=5632
export NUM_HEADS=16
export NUM_KV_HEADS=8
export LINEAR_KEY_HEAD_DIM=64
export LINEAR_VALUE_HEAD_DIM=64
export LINEAR_NUM_KEY_HEADS=8
export LINEAR_NUM_VALUE_HEADS=16
# Pure (all-linear) 24-layer stack. For a hybrid GDN/SDPA stack instead, set '3' (1 SDPA every 3).
export LINEAR_ATTENTION_FREQ=${LINEAR_ATTENTION_FREQ:-[1]*24}

# ---- 40B tokens: GBS 512 (= 2.10M tokens/step) x 19073 steps; seq 4096 ----
export MBS=${MBS:-2}
export GBS=${GBS:-512}
export SEQ_LEN=4096
export TRAIN_SAMPLES=9765625          # 9,765,625 * 4096 = 40.0B tokens
export MUON_LR=${MUON_LR:-3.6e-4}      # 350M used 3.6e-4; lower if 1.3B is unstable
export MIN_LR=${MIN_LR:-3.6e-5}

# ---- checkpointing (so the run resumes if it hits the walltime) ----
export ENABLE_CHECKPOINTING=1
export SAVE_INTERVAL=${SAVE_INTERVAL:-500}
export CKPT_FORMAT=torch
export SAVE_DIR=${SAVE_DIR:-/iopsstor/scratch/cscs/$USER/linear-attention-architectures/_research/results/checkpoints/${EXP_NAME:-run}-${SLURM_JOB_ID:-manual}}
export EVAL_INTERVAL=${EVAL_INTERVAL:-200}
export EVAL_ITERS=${EVAL_ITERS:-10}

# ---- multi-node safety (avoid NCCL init hang / Inductor fuser issues at scale) ----
export MEGATRON_DISABLE_JIT_FUSER=1
export MEGATRON_INIT_PROCESS_GROUP_DEVICE_ID=1
export MEGATRON_LINEAR_TORCH_COMPILE=1
export MEGATRON_LINEAR_TORCH_COMPILE_MODE=reduce-overhead
export MEGATRON_LINEAR_TORCH_COMPILE_CUDAGRAPHS=0

cd "$CLER_REPO_ROOT"
exec bash _research/launch/transformer-pp-350m-linear-muon.sbatch
