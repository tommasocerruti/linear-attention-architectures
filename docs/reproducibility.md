# Reproducing CLER Results

This document covers everything needed to reproduce the training runs and
downstream evaluations from the CLER paper on the Clariden (Swiss AI Alps)
cluster at CSCS. The steps proceed in order: cluster access → environment →
dataset → training → evaluation.

---

## 1. Prerequisites

### 1.1 Cluster access

All experiments run on **Clariden**, the Swiss AI partition of the Alps
supercomputer at CSCS (Swiss National Supercomputing Centre). You need:

- A CSCS account with access to the `infra01` (or `lsaie-ss26`) project.
- SSH keys via the CSCS key service (`ela.cscs.ch`).

Log in:

```bash
./cscs-keygen.sh          # refresh your 24h CSCS key
ssh clariden              # proxies through ela.cscs.ch
```

### 1.2 Node hardware

Each Clariden compute node has:

| Resource | Specification |
| --- | --- |
| GPU | 4× NVIDIA GH200 Grace-Hopper (ARM + GPU on one chip) |
| GPU memory | 96 GB HBM3 per GPU |
| CPU | ARM Neoverse V2, 72 cores per GPU process |
| Host memory | 480 GB LPDDR5X |
| Interconnect | Slingshot-11 (200 Gb/s per port) |
| Architecture | AArch64 (ARM 64-bit) — **not x86_64** |

Training runs use **2 nodes (8 GH200 GPUs total)**.
Evaluation runs use **1 node (4 GPUs)**.

### 1.3 Container

All compute jobs run inside the official Alps3 PyTorch container:

```
jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.01-py3-alps3
```

This is declared in `_research/launch/alps3.toml`. No manual container build
is required; the `.toml` is passed directly to `srun --environment=`.

### 1.4 Persistent storage layout

| Path | Purpose |
| --- | --- |
| `/iopsstor/scratch/cscs/$USER/` | Fast Lustre scratch (write from compute nodes) |
| `/capstor/scratch/cscs/$USER/` | Slower long-term storage (do not write from jobs) |
| `/users/$USER/` | Home directory (shared across login and compute) |

The repository on login nodes (`/users/<user>/cler`) is symlinked to
`/iopsstor/scratch/cscs/<user>/cler` so that compute nodes can reach the same
code via the fast scratch path.

---

## 2. Repository Setup

```bash
# Clone the repository (replace with the public arXiv release URL)
git clone <repo-url> ~/cler
cd ~/cler

# Symlink so compute nodes can find it via iopsstor
ln -sfn /users/$USER/cler /iopsstor/scratch/cscs/$USER/cler
```

The main branch for training is `mega-cler`. Check it out:

```bash
git checkout mega-cler
```

Create output directories on scratch:

```bash
mkdir -p /iopsstor/scratch/cscs/$USER/cler/_research/results/{runs,checkpoints,eval}
mkdir -p /iopsstor/scratch/cscs/$USER/cler/cache/{tmp,triton,inductor}
```

### 2.1 Python dependencies

Dependencies are installed automatically at job startup by
`_research/launch/install_python_deps.sh`. This script runs inside the
container and installs packages into a target directory on scratch, so **no
manual `pip install` is required** before submitting jobs.

Key packages installed at runtime:

| Package | Version | Purpose |
| --- | --- | --- |
| `flash-linear-attention` | 0.5.0 | FLA Triton kernels for GDN/DeltaNet |
| `fla-core` | 0.5.0 | Core FLA ops (`chunk_delta_rule`, `chunk_gated_delta_rule`) |
| `causal-conv1d` | ~1.5 | Short depthwise convolution for token mixing |
| `tilelang` | latest | Tile-based language for custom GPU kernels |
| `transformers`, `datasets`, `wandb` | latest | HF stack and logging |
| NVIDIA Emerging Optimizers | v0.2.0 | Muon optimizer (`adaptive_muon`/`normuon`) |

These install to `_research/packages-server-py311/` (or `py36` for older
Python login nodes) and are skipped on subsequent jobs if a marker file is
present.

---

## 3. Dataset

### 3.1 Source

Training uses [FineWeb-Edu 100BT shuffled](https://huggingface.co/datasets/HuggingFaceFW/fineweb_edu_100BT-shuffled)
tokenised with the **LLaMA-2 SentencePiece tokenizer** (32 000-token
vocabulary, `tokenizer.model` from Meta).

### 3.2 Megatron binary format

`lm_eval` datasets need no preprocessing. The training data must be converted
to Megatron's binary format. The 62B-token `_tc` prefix used in all final runs
is located at:

```
/iopsstor/scratch/cscs/lingfeng/cler/_research/results/data/fineweb_edu/
  fineweb_edu_62b_llama2_tc/
    fineweb_edu_62b_llama2_tc_text_document.bin
    fineweb_edu_62b_llama2_tc_text_document.idx
```

To reproduce the preprocessing yourself:

```bash
python tools/preprocess_data.py \
    --input <fineweb_edu_jsonl_path> \
    --output-prefix /iopsstor/scratch/cscs/$USER/cler/_research/results/data/fineweb_edu_llama2 \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --workers 32 \
    --chunk-size 1024
```

### 3.3 Tokenizer

The LLaMA-2 tokenizer model file is at:

```
/iopsstor/scratch/cscs/lingfeng/llama2-tokenizer/tokenizer.model
```

Set this path in `LLAMA2_TOKENIZER_MODEL` before running any job.

---

## 4. Training

### 4.1 Architecture

All final runs use the same **350M-class hybrid** architecture:

| Hyperparameter | Value |
| --- | --- |
| Parameters | ~350M |
| Layers | 20 |
| Hidden size | 1024 |
| FFN hidden size | 2816 |
| Attention heads | 16 |
| KV heads (GQA) | 4 |
| Sequence length | 4096 |
| Position embedding | RoPE (base 500 000) |
| Normalization | RMSNorm |
| Activation | SwiGLU |
| Linear/attention mix | `linear_attention_freq=3` (3 linear layers per attention layer) |
| Precision | bf16 |

### 4.2 Optimizer

All runs use the **Muon** optimizer (`adaptive_muon` / `normuon`) with the
following settings:

| Setting | Value |
| --- | --- |
| Muon LR | 3.6e-4 |
| Scalar LR | 1.5e-3 |
| Momentum | 0.95 (Nesterov) |
| Scale mode | spectral |
| LR schedule | WSD (warmup–stable–decay, `minus_sqrt` decay) |
| Warmup samples | 25 600 |
| WSD decay samples | 732 422 |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| Global batch size | 128 |
| Micro batch size | 2 |

### 4.3 Training budget

| Setting | Value |
| --- | --- |
| Training samples | 3 662 109 |
| Training tokens | ~15B |
| Iterations | 28 610 |
| Checkpoint interval | 500 iterations |

### 4.4 Linear attention variants

The `LINEAR_VARIANT` environment variable selects the architecture. Each
variant maps to specific Megatron CLI flags (see
`_research/launch/transformer-pp-350m-linear-muon.sbatch` for the full
mapping):

| `LINEAR_VARIANT` | Backbone | CLER | `experimental_attention_variant` |
| --- | --- | --- | --- |
| `gdn_triton` | Gated DeltaNet (FLA Triton) | off | `gated_delta_net` |
| `cler_gdn_triton` | Gated DeltaNet (FLA Triton) | on | `gated_delta_net` |
| `delta_net` | DeltaNet (FLA Triton) | off | `delta_net` |
| `cler_deltanet_triton` | DeltaNet (FLA Triton) | on | `delta_net` |

CLER variants additionally pass:

| Flag | CLER-H | CLER-V |
| --- | --- | --- |
| `--cler-enabled` | ✓ | ✓ |
| `--cler-hidden-routing` | ✓ | ✓ |
| `--cler-hidden-route-value` | — | ✓ |
| `--cler-gamma-mode` | `scalar` | `scalar` |
| `--cler-routing-mode` | `latest` | `latest` |

### 4.5 Launching the final runs

Each final experiment has a self-contained launcher in `_research/launch/`.
Before running, set your W&B API key and update the data/tokenizer paths if
they differ from Lingfeng's paths.

```bash
# Example: GDN baseline
sbatch --account=infra01 _research/launch/final-GDN-350M-15B.sbatch

# CLER-H
sbatch --account=infra01 _research/launch/final-CLER-H-350M-15B.sbatch

# CLER-V
sbatch --account=infra01 _research/launch/final-CLER-V-350M-15B.sbatch

# DeltaNet baseline
sbatch --account=infra01 _research/launch/final-DELTANET-350M-15B.sbatch

# DeltaNet CLER-H
sbatch --account=infra01 _research/launch/final-DELTANET-CLER-H-350M-15B.sbatch

# DeltaNet CLER-V
sbatch --account=infra01 _research/launch/final-DELTANET-CLER-V-350M-15B.sbatch
```

Each launcher `exec`s into `transformer-pp-350m-linear-muon.sbatch` with the
appropriate `LINEAR_VARIANT` and CLER env vars set.

SLURM allocation per job: **2 nodes, 4 GPUs each, 12-hour wall time**.

Training logs go to:

```
/iopsstor/scratch/cscs/$USER/cler/_research/results/runs/<JOB_NAME>-<JOB_ID>.log
```

### 4.6 Checkpoints

Checkpoints are saved in Megatron `torch` format every 500 iterations to:

```
/iopsstor/scratch/cscs/$USER/cler/_research/results/checkpoints/<EXP_NAME>/
```

Only the latest checkpoint is kept by default (`SAVE_RETAIN_INTERVAL=29000`
ensures the final checkpoint is always preserved). Each checkpoint is 3–4 GB.

The pre-trained checkpoints from our runs are stored at:

```
/iopsstor/scratch/cscs/lingfeng/cler/_research/results/checkpoints/
  GDN-350M-15B-MUON/iter_0028610/
  CLER-H-350M-15B-MUON/iter_0028610/
  CLER-V-350M-15B-MUON/iter_0028610/
  DELTANET-350M-15B-MUON/iter_0028610/
  DELTANET-CLER-H-350M-15B-MUON/iter_0028610/
  DELTANET-CLER-V-350M-15B-MUON/iter_0028610/
```

> **Note on scratch retention**: `iopsstor` scratch is subject to CSCS cleanup
> policies. If you plan to use these checkpoints weeks after the paper's
> publication, copy them to `/capstor` using the `xfer` partition (not the
> login node).

---

## 5. Downstream Evaluation

Downstream evaluations use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
running against a local Megatron scoring server. This avoids any HuggingFace
checkpoint conversion.

### 5.1 How it works

```
lm_eval (host process)
    │  /v1/completions  (loglikelihood requests)
    ▼
run_loglikelihood_scoring_server.py  (inside ALPS3 container)
    │  full-sequence forward pass  (no autoregressive cache)
    ▼
Megatron model loaded from native checkpoint
```

`lm_eval` sends each continuation as an API request. The scoring server
computes the full-sequence loglikelihood in a single forward pass, which
is required for DeltaNet and GDN variants that do not support Megatron's
incremental autoregressive inference.

### 5.2 Key scripts

| Script | Role |
| --- | --- |
| `_research/eval/run_lm_eval_local_api.sbatch` | SLURM wrapper: starts server, waits for readiness, runs `lm_eval` |
| `tools/run_loglikelihood_scoring_server.py` | Scoring server (full-sequence loglikelihood, no KV-cache) |
| `_research/eval/pythonpath/sitecustomize.py` | Patches `lm_eval`'s `RemoteTokenizer` to cache and pace API calls |

### 5.3 Submitting an evaluation job

```bash
CKPT_ROOT=/iopsstor/scratch/cscs/lingfeng/cler/_research/results/checkpoints
TOKENIZER=/iopsstor/scratch/cscs/lingfeng/llama2-tokenizer/tokenizer.model
MODEL=GDN-350M-15B-MUON   # replace with desired model name

sbatch \
  --account=infra01 \
  --time=01:30:00 \
  --job-name="${MODEL}-lm-eval" \
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

SLURM allocation per eval job: **1 node, 4 GPUs, 90-minute wall time**.

### 5.4 Eval settings

| Setting | Value |
| --- | --- |
| Tasks | HellaSwag, PIQA, WinoGrande |
| Metric | `acc` and `acc_norm` (length-normalised) |
| Batch size | 8 |
| Max sequence length | 4096 |
| Tokenizer | LLaMA-2 SentencePiece (`Llama2Tokenizer`) |
| Server mode | `scoring` (full-sequence, no autoregressive cache) |
| `--transformer-impl` | `transformer_engine` |
| `--attention-backend` | `unfused` |

### 5.5 Output location

Results land at:

```
/iopsstor/scratch/cscs/$USER/cler/_research/results/eval/<RUN_NAME>/cler/results_*.json
```

The server startup log (useful for verifying checkpoint args were restored):

```
/iopsstor/scratch/cscs/$USER/cler/_research/results/eval/<RUN_NAME>-server-<JOB_ID>.log
```

### 5.6 Verifying that model architecture args were restored

After the server starts, grep the server log to confirm the architecture and
CLER flags were correctly restored from the checkpoint:

```bash
grep -E "Setting (experimental_attention_variant|linear_attention_freq|cler_enabled|cler_hidden_routing|cler_hidden_route_value) to" \
  /iopsstor/scratch/cscs/$USER/cler/_research/results/eval/<RUN_NAME>-server-<JOB_ID>.log
```

Expected output for CLER-V:

```
Setting experimental_attention_variant to gated_delta_net from checkpoint
Setting linear_attention_freq to 3 from checkpoint
Setting cler_enabled to True from checkpoint
Setting cler_hidden_routing to True from checkpoint
Setting cler_hidden_route_value to True from checkpoint
```

If any of these lines are missing the model may be running with wrong
architecture settings. Do not trust scores from a server log that does not
show all expected `Setting ... from checkpoint` lines.

### 5.7 Smoke test (optional, fast)

Before a full eval, run a 10-sample smoke test on the `debug` partition:

```bash
sbatch \
  --account=infra01 \
  --partition=debug \
  --time=00:15:00 \
  --job-name="${MODEL}-smoke" \
  --export=ALL,\
CKPT=$CKPT_ROOT/$MODEL,\
LLAMA2_TOKENIZER_MODEL=$TOKENIZER,\
RUN_NAME=${MODEL}-smoke10,\
TASKS=hellaswag,\
SERVER_IMPL=scoring,\
LIMIT=10,\
SERVER_EXTRA_ARGS="--transformer-impl transformer_engine --attention-backend unfused --no-persist-layer-norm --no-use-tokenizer-model-from-checkpoint-args --tokenizer-type Llama2Tokenizer --tokenizer-model $TOKENIZER",\
LM_EVAL_BATCH_SIZE=8,\
LM_EVAL_MAX_LENGTH=4096,\
HF_DATASETS_CACHE=/iopsstor/scratch/cscs/$USER/hf_cache/datasets \
  _research/eval/run_lm_eval_local_api.sbatch
```

Tail the job log to monitor:

```bash
tail -f /iopsstor/scratch/cscs/$USER/cler/_research/results/eval/<RUN_NAME>-smoke10-<JOB_ID>.log
```

---

## 6. Expected Results

### 6.1 Final validation loss (iter 28 610 = 15B tokens)

| Model | Final val loss | Δ vs baseline |
| --- | ---: | ---: |
| GDN-350M-15B-MUON | 2.3417 | — |
| CLER-H-350M-15B-MUON | 2.3375 | −0.0042 |
| CLER-V-350M-15B-MUON | 2.3358 | −0.0059 |
| DELTANET-350M-15B-MUON | 2.3347 | — |
| DELTANET-CLER-H-350M-15B-MUON | 2.3345 | −0.0002 |
| DELTANET-CLER-V-350M-15B-MUON | 2.3331 | −0.0016 |

### 6.2 Downstream tasks (HellaSwag / PIQA / WinoGrande, iter 28 610)

Results below are from the corrected eval (mega-cler branch, CLER args
properly restored from checkpoint). GDN and DeltaNet baselines from
initial eval (same settings, architecture args confirmed in server logs).

*(Table will be updated once the CLER rerun jobs complete.)*

| Model | HellaSwag acc_norm | PIQA acc_norm | WinoGrande acc |
| --- | ---: | ---: | ---: |
| GDN-350M-15B-MUON | 0.4131 | 0.6649 | 0.5146 |
| DELTANET-350M-15B-MUON | 0.4113 | 0.6605 | 0.5193 |
| CLER-H-350M-15B-MUON | *pending* | *pending* | *pending* |
| CLER-V-350M-15B-MUON | *pending* | *pending* | *pending* |
| DELTANET-CLER-H-350M-15B-MUON | *pending* | *pending* | *pending* |
| DELTANET-CLER-V-350M-15B-MUON | *pending* | *pending* | *pending* |

---

## 7. Troubleshooting

**Server never becomes ready** — inspect the server log first:

```bash
tail -n 200 /iopsstor/scratch/cscs/$USER/cler/_research/results/eval/<RUN_NAME>-server-<JOB_ID>.log
```

**Bus error (core dumped) during dataset loading** — HuggingFace datasets
must not write Arrow cache to the NFS home directory. Always set:

```bash
export HF_DATASETS_CACHE=/iopsstor/scratch/cscs/$USER/hf_cache/datasets
export HF_HOME=/iopsstor/scratch/cscs/$USER/hf_home
```

**Architecture args not restored (e.g. `experimental_attention_variant=None`)** —
ensure `--use-checkpoint-args` is passed to the server (it is, by default, in
the sbatch). Check that the `mega-cler` branch is checked out; the main branch
is missing the `_set_arg()` calls for linear attention and CLER args in
`megatron/training/checkpointing.py`.

**CLER-V scores collapse to near-chance** — CLER was not active during eval.
Confirm `Setting cler_hidden_route_value to True from checkpoint` appears in
the server log. If missing, see the previous item.

**`DeltaNet does not support inference for now`** — you are using the
autoregressive `run_text_generation_server.py` instead of the scoring server.
Always use `SERVER_IMPL=scoring` for DeltaNet and GDN variants.

**`ModuleNotFoundError: megatron.core.parallel_state`** — the local repo is
not first on `PYTHONPATH`. The sbatch sets this correctly; ensure you are not
overriding `PYTHONPATH` in your shell environment.
