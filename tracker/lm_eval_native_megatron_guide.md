# Native Megatron lm_eval Guide

This flow runs `lm_eval` against checkpoints that were written by Megatron
itself. It avoids Hugging Face loading and conversion: `lm_eval` remains the
outer harness, while a local Megatron text-generation server loads the native
checkpoint and exposes the OpenAI-style completion and remote-tokenizer APIs
that `lm_eval` needs.

Use this path for Transformer++ and nearby native Megatron variants unless the
checkpoint has first been converted to a Hugging Face format.

## Components

- `_research/eval/run_lm_eval_local_api.sbatch` starts the local server, waits
  for readiness, activates the eval environment, and runs `lm_eval`.
- `tools/run_text_generation_server.py` is the static server entrypoint. The
  working path uses `--transformer-impl local`, `--attention-backend unfused`,
  and `--no-persist-layer-norm`.
- `megatron/core/inference/text_generation_server/text_generation_server.py`
  exposes `/v1/completions`, `/v1/tokenizer_info`, `/v1/tokenize`, and
  `/v1/detokenize`.
- `megatron/core/inference/text_generation_server/tokenizer_api.py` keeps the
  tokenizer endpoint behavior small and testable.
- `_research/eval/pythonpath/sitecustomize.py` patches `lm_eval`'s
  `RemoteTokenizer` inside the eval process to cache and pace tokenizer calls.
- `megatron/__init__.py` keeps the checkout's `megatron` tree a regular
  package so server-side package directories cannot shadow local modules.

## Full Eval Pattern

Set the tokenizer, server flags, tasks, and native checkpoint root, then submit:

```bash
cd /iopsstor/scratch/cscs/course_00206/cler

export LLAMA2_TOKENIZER_MODEL=/iopsstor/scratch/cscs/course_00206/llama2-tokenizer/tokenizer.model
export SERVER_EXTRA_ARGS="--transformer-impl local --attention-backend unfused --no-persist-layer-norm"
export TASKS='hellaswag,piqa,winogrande'
unset LIMIT

CKPT=/absolute/path/to/native-megatron-checkpoint-dir
RUN_NAME=my-model-lm-eval-full-3tasks

JID=$(sbatch --parsable \
  --partition=normal \
  --time=02:00:00 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=1 \
  --cpus-per-task=72 \
  --mem=230000 \
  --job-name=eval-my-model-full \
  --export=ALL,SERVER_IMPL=static,CONDA_ENV_NAME=cler-eval,CKPT=$CKPT,RUN_NAME=$RUN_NAME,READY_MAX_TRIES=36,READY_SLEEP_SECONDS=5 \
  _research/eval/run_lm_eval_local_api.sbatch)

echo "$JID"
```

The output JSON lands under:

```text
_research/results/eval/$RUN_NAME/cler/results_*.json
```

## Smoke First

Before a full run, set a small limit and a short debug allocation:

```bash
export LIMIT=10
export TASKS='hellaswag,piqa,winogrande'

JSMOKE=$(sbatch --parsable \
  --partition=debug \
  --time=00:15:00 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=1 \
  --cpus-per-task=72 \
  --mem=230000 \
  --job-name=eval-my-model-smoke10 \
  --export=ALL,SERVER_IMPL=static,CONDA_ENV_NAME=cler-eval,CKPT=$CKPT,RUN_NAME=$RUN_NAME-smoke10,READY_MAX_TRIES=36,READY_SLEEP_SECONDS=5 \
  _research/eval/run_lm_eval_local_api.sbatch)
```

Watch the outer log for harness progress:

```bash
tail -f _research/results/eval/eval-my-model-smoke10-$JSMOKE.log
```

Use the server log as the primary startup signal:

```bash
tail -n 120 _research/results/eval/$RUN_NAME-smoke10-server-$JSMOKE.log
```

## Adapting A New Implementation

Keep the outer harness unchanged first. Add or select the model implementation
so `tools/run_text_generation_server.py --use-checkpoint-args --load $CKPT`
can build the model from native Megatron checkpoint args. Start with:

```bash
export SERVER_EXTRA_ARGS="--transformer-impl local --attention-backend unfused --no-persist-layer-norm"
```

Only change the server flags after the local static path is healthy. If a new
implementation needs optional imports or kernels, make those imports lazy or
guarded so the standard Transformer++ path still starts.

## Do Not Change Casually

- Do not use `_research/eval/run_lm_eval_hf.sh` for native Megatron checkpoints
  unless the checkpoint has been converted first.
- Do not switch TPP smoke/full evals back to the dynamic server while the static
  path works.
- Do not remove the remote tokenizer endpoints; `lm_eval` uses them for
  `local-completions` loglikelihood requests.
- Do not remove `megatron/__init__.py` unless import diagnostics prove package
  shadowing is impossible in the server environment.
- Do not put eval-side package directories ahead of the repo on the server
  `PYTHONPATH`.

## Troubleshooting

- `ModuleNotFoundError: megatron.core.parallel_state`: check `sys.path` and
  `importlib.util.find_spec('megatron.core.parallel_state')`. The local repo
  must be first, and `megatron/__init__.py` must be present.
- Server never becomes ready: inspect the `*-server-$SLURM_JOB_ID.log` first.
  Most failures happen before `lm_eval` starts.
- `lm_eval` tokenizer calls fail with connection exhaustion or
  `Cannot assign requested address`: keep `_research/eval/pythonpath` on the
  eval-side `PYTHONPATH` so the `RemoteTokenizer` cache/pacing patch loads.
- `/v1/completions` returns validation errors for loglikelihood requests:
  verify the stale top-k guard has not been restored in
  `endpoints/completions.py`.
- Optional linear-attention imports break TPP startup: keep optional model
  imports guarded so importing `gpt_builders.py` does not require unavailable
  kernels for the standard Transformer++ path.
