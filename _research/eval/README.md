# Evaluation

Paper downstream evaluation uses `lm-eval` against a local Megatron scoring server. This avoids converting checkpoints to Hugging Face format and works for DeltaNet/Gated DeltaNet variants that do not use Megatron's autoregressive KV-cache path.

## Entrypoints

| File | Role |
| --- | --- |
| `run_lm_eval_local_api.sbatch` | SLURM wrapper that starts the server, waits for readiness, and runs `lm-eval` |
| `tools/run_loglikelihood_scoring_server.py` | Full-sequence loglikelihood server for native Megatron checkpoints |
| `pythonpath/sitecustomize.py` | Small `lm-eval` API-tokenizer patch for caching and pacing requests |

## Typical Run

```bash
CKPT_ROOT=/path/to/checkpoints
TOKENIZER=/path/to/tokenizer.model
MODEL=GDN-350M-15B-MUON

sbatch \
  --export=ALL,\
CKPT=$CKPT_ROOT/$MODEL,\
LLAMA2_TOKENIZER_MODEL=$TOKENIZER,\
RUN_NAME=${MODEL}-lm-eval,\
TASKS=hellaswag,piqa,winogrande,\
SERVER_IMPL=scoring,\
SERVER_EXTRA_ARGS="--transformer-impl transformer_engine --attention-backend unfused --no-persist-layer-norm --no-use-tokenizer-model-from-checkpoint-args --tokenizer-type Llama2Tokenizer --tokenizer-model $TOKENIZER",\
LM_EVAL_BATCH_SIZE=8,\
LM_EVAL_MAX_LENGTH=4096,\
HF_DATASETS_CACHE=/iopsstor/scratch/cscs/$USER/hf_cache/datasets,\
HF_HOME=/iopsstor/scratch/cscs/$USER/hf_home \
  _research/eval/run_lm_eval_local_api.sbatch
```

Before trusting a score, inspect the server log and confirm that checkpoint arguments restored the expected `experimental_attention_variant`, `linear_attention_freq`, and CLER flags.
