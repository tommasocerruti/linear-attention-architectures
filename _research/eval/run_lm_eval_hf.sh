#!/bin/bash
set -euo pipefail

MODEL_PATH=${1:?usage: run_lm_eval_hf.sh HF_MODEL_OR_CHECKPOINT_PATH [TASKS]}
TASKS=${2:-wikitext,lambada_openai}

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PACKAGE_DIR=${EVAL_PACKAGE_DIR:-$REPO_DIR/_research/packages-eval}
OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-$REPO_DIR/_research/results/eval}
RUN_NAME=$(basename "$MODEL_PATH")
OUTPUT_DIR=$OUTPUT_ROOT/$RUN_NAME

mkdir -p "$PACKAGE_DIR" "$OUTPUT_DIR"

if command -v uv >/dev/null 2>&1; then
    INSTALL="uv pip install --quiet"
else
    INSTALL="pip install --quiet"
fi

if ! PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 -c "import lm_eval" >/dev/null 2>&1; then
    $INSTALL --target="$PACKAGE_DIR" 'lm_eval[hf]'
fi

export PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$TASKS" \
    --batch_size auto \
    --output_path "$OUTPUT_DIR"
