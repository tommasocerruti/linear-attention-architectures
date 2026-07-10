# Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing

<p align="center">
  <a href="https://arxiv.org/abs/2607.07953"><img src="https://img.shields.io/badge/arXiv-2607.07953-b31b1b.svg" alt="arXiv"></a>
  <a href="https://arxiv.org/pdf/2607.07953"><img src="https://img.shields.io/badge/Paper-PDF-blue.svg" alt="Paper PDF"></a>
</p>

This repository accompanies the technical report [Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing](https://arxiv.org/abs/2607.07953).

If you use this repository or build on the accompanying report, please cite:

```bibtex
@misc{cerruti2026linearattentionarchitecturesmechanisms,
      title={Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing}, 
      author={Tommaso Cerruti and Tim Rieder and George Rowlands and Lingfeng Jin and Imanol Schlag},
      year={2026},
      eprint={2607.07953},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.07953}, 
}
```

## When to cite

Consider citing this report if your work compares modern linear-attention architectures, uses DeltaNet/Gated DeltaNet/Kimi Delta Attention/Gated DeltaNet-2 baselines, studies recurrent associative memory for language models, evaluates hybrid versus pure linear-attention stacks, or builds on CLER/CLVR-style cross-layer routing.

It is a fork of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), via the [Clariden research baseline](https://github.com/ischlag/megatron-lm-research-baseline), used to train and evaluate linear-attention language models at the 350M–3B parameter scale.

## Mechanisms supported

The code supports the mechanisms studied in the report:

- DeltaNet
- Gated DeltaNet
- Kimi Delta Attention
- Gated DeltaNet-2
- CLER (Cross-Layer Error Residuals), cross-layer error routing
- CLVR (Cross-Layer Value Routing), routing write values into the residual stream for DeltaNet-style architectures

The main implementation lives under `megatron/core/ssm/` and the routing state is integrated through `megatron/core/transformer/transformer_block.py`. The launch scripts under `_research/launch/` are the executable source of truth for experiment-specific flags.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `megatron/` | Megatron core plus the linear-attention and routing implementations |
| `_research/data/` | FineWeb-Edu preprocessing entrypoints and tokenizer assets used by the experiments |
| `_research/launch/` | SLURM launchers for final runs, scale runs, ablations, and smoke checks |
| `_research/eval/` | Downstream `lm-eval` wrappers for native Megatron checkpoints |
| `tools/run_loglikelihood_scoring_server.py` | Full-sequence scoring server used by downstream evaluation |
| `tests/unit_tests/ssm/` | Focused CPU tests for linear-attention and CLER routing logic |
| `docs/reproducibility.md` | Longer reproduction notes for Clariden |

## Environment

The production launchers target Clariden GH200 nodes through SLURM and the Alps3 PyTorch container declared in `_research/launch/alps3.toml`. Most scripts can be moved to another cluster, but the `#SBATCH` headers, container path, scratch paths, and data locations are site-specific.

Set these variables before launching training or evaluation:

```bash
export CLER_REPO_ROOT=/iopsstor/scratch/cscs/$USER/linear-attention-architectures
export MEGATRON_DATA_PATH=/path/to/fineweb_edu_62b_llama2_tc_text_document
export LLAMA2_TOKENIZER_MODEL=/path/to/tokenizer.model
export WANDB_API_KEY=<optional-wandb-key>
export WANDB_PROJECT=<optional-wandb-project>
export HF_HOME=/iopsstor/scratch/cscs/$USER/hf_home
export HF_DATASETS_CACHE=/iopsstor/scratch/cscs/$USER/hf_cache/datasets
```

`MEGATRON_DATA_PATH` is the Megatron binary prefix without `.bin` or `.idx`. The final paper launchers default to the paths used on Clariden; override the variables above for a fresh checkout.

## Data Preparation

For a quick conversion check:

```bash
sbatch _research/data/convert_fineweb_edu_smoke.sbatch
```

For the full FineWeb-Edu/LLaMA-2 preprocessing path:

```bash
sbatch _research/data/convert_fineweb_edu.sbatch
```

The conversion scripts prepare FineWeb-Edu text into Megatron binary format with the LLaMA-2 SentencePiece tokenizer. See `_research/data/README.md` and `docs/reproducibility.md` for cluster paths and token-count variants.

## Training

Final 350M/15B-token runs are launched through the `final-*.sbatch` wrappers:

```bash
sbatch _research/launch/final-GDN-350M-15B.sbatch
sbatch _research/launch/final-CLER-H-350M-15B.sbatch
sbatch _research/launch/final-CLER-V-350M-15B.sbatch
sbatch _research/launch/final-DELTANET-350M-15B.sbatch
sbatch _research/launch/final-DELTANET-CLER-H-350M-15B.sbatch
sbatch _research/launch/final-DELTANET-CLER-V-350M-15B.sbatch
```

Those wrappers set model-specific environment variables and then execute `_research/launch/transformer-pp-350m-linear-muon.sbatch`, which pins the shared architecture, optimizer, schedule, checkpointing, and data flags.

Other retained paper entrypoints:

| Purpose | Entrypoint |
| --- | --- |
| 350M LR ablations | `_research/launch/submit-lr-ablation-2000-v2.sh` |
| Sequence-length scaling | `_research/launch/submit-attn-dominated-gdn-seq-scaling.sh` |
| 1.3B / 40B routing scale runs | `_research/launch/transformer-pp-1p3b-fwe40b-*.sbatch` |
| 3B DeltaNet scale runs | `_research/launch/transformer-pp-3b-deltanet-muon.sbatch` |
| Gated DeltaNet-2 scale probe | `_research/launch/transformer-pp-1p3b-fwe15b-gdn2-pytorch-adamw-17h58-external.sbatch` |

Use smoke launchers before expensive runs when changing environments. Useful retained checks include `_research/launch/fla-import-smoke.sbatch`, `_research/launch/transformer-pp-350m-gdn-pytorch-smoke.sbatch`, `_research/launch/transformer-pp-350m-deltanet-smoke.sbatch`, `_research/launch/transformer-pp-350m-gdn2-smoke.sbatch`, and `_research/launch/transformer-pp-350m-kda-1gpu-smoke.sbatch`.

## Evaluation

Downstream scores are produced with `lm-eval` against a local full-sequence Megatron scoring server. This keeps evaluation on native Megatron checkpoints and avoids Hugging Face conversion.

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
HF_DATASETS_CACHE=$HF_DATASETS_CACHE,\
HF_HOME=$HF_HOME \
  _research/eval/run_lm_eval_local_api.sbatch
```

The wrapper starts `tools/run_loglikelihood_scoring_server.py`, waits for readiness, and then runs `lm-eval`. Use the server log to confirm that linear-attention and CLER checkpoint arguments were restored before trusting scores.
