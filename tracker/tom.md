# Tommaso

## Week 1

**Goal**: Working Clariden pipeline for an initial architecture comparison.

### Main Experimental Choices
Choices taken to be coherent with the core Gated DeltaNet implementation.
- **Optimizer:** AdamW. Later, we will try Muon (Ling already did).
- **Dataset:** FineWeb-Edu. Incrementally increase the dataset size 50M (quick run) -> 15B (initial baseline) -> 100B (final baseline).
- **Tokenizer:** LLaMA-2 SentencePiece.
- **Cluster setting:** Clariden, one GH200 node with 4 GPUs, using pure
  data-parallel training with tensor, pipeline, and context parallelism all
  kept at 1.

### Clariden Validation Completed
The full path was validated on Clariden in the intended runtime environment:
- FineWeb-Edu quick conversion completed successfully (about **50M tokens**, 99% for training and 1% for validation).
- L (#layers) was kept fixed for the Transformer++ reference and adjusted for Gated DeltaNet and DeltaNet so that parameter counts stayed within about +/- 2% of the Transformer++ baseline.

The current quick-run comparison is:

| Baseline | Params | Optimizer | Data | Tokenizer | Final train loss | Final val loss | Val PPL | Throughput (TFLOP/s/GPU) | Throughput (ktokens/s/GPU, approx.) |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Transformer++ softmax (24L) | 317.2M | AdamW | FineWeb-Edu 50M | LLaMA-2 | 5.9744 | 6.0082 | 406.8 | 331.0 | 171 |
| Gated DeltaNet (22L) | 313.4M | AdamW | FineWeb-Edu 50M | LLaMA-2 | 5.1020 | 5.1217 | 167.6 | 302.0 | 156 |
| DeltaNet (24L) | 319.5M | AdamW | FineWeb-Edu 50M | LLaMA-2 | 6.2663 | 6.3069 | 548.3 | 294.3 | 152 |

`lm loss` is the autoregressive cross-entropy / negative log-likelihood
averaged over predicted tokens, and perplexity is therefore `exp(loss)`.
The `ktokens/s/GPU` values are approximate for softmax and Gated DeltaNet. I
used the measured DeltaNet value (`152,136.8 tokens/s/GPU`) and scaled it by
the throughput ratios:
`tokens/s ≈ 152,136.8 * (throughput_model / 294.3)`, then rounded to the
nearest integer in `ktokens/s/GPU`.

The figure below shows the same three-way comparison visually: training loss,
validation loss, and throughput over the near-full 50M-token run.

Open full-size image: [clariden_baseline_comparison_2026-04-23.png](week1/clariden_baseline_comparison_2026-04-23.png)

![Near-full 50M baseline comparison](week1/clariden_baseline_comparison_2026-04-23.png)

Operationally, this establishes that:

- the data path works,
- the tokenizer path works,
- the Clariden sbatches work,
- all three baseline training pipelines run successfully on one GH200 node.

### Interpretation

The initial quick runs were operational and numerical checks rather than final
benchmark runs. Their purpose was to confirm that:

- the FineWeb-Edu data conversion is valid,
- the LLaMA-2 tokenizer is loaded correctly,
- the Megatron binary dataset is readable,
- the model builds successfully on Clariden,
- distributed training runs on 4 GH200 GPUs,
- validation executes correctly,
- W&B logging works,
- training remains numerically stable without NaNs.

At this stage, the main picture is:
- **Gated DeltaNet** is best on quality,
- **Transformer++ softmax** is best on raw throughput,
- **plain DeltaNet** is currently the weakest of the three and is most useful
  as an ungated ablation.

The plain DeltaNet result is important because it makes the role of gating
much clearer. In this setup, the gating in GDN is doing substantial work
rather than acting as a small implementation detail.

### Next Week

1. Launch longer runs (1B-tokens and 15B-tokens dataset versions of FineWeb-edu) for:
   - 350M Transformer++ + AdamW
   - 350M Gated DeltaNet + AdamW
   - 350M DeltaNet + AdamW
   - 350M Transformer++ + Muon
   - 350M Gated DeltaNet + Muon
   - 350M DeltaNet + Muon
2. Compare the longer runs on the same metrics as above (loss, perplexity, and throughput).  
3. Have a stable baseline comparison to start to integrate CLER.

### Note For Myself: Baseline Implementation Details

Not for discussion unless needed.

What I implemented:
- Added a FineWeb-Edu conversion path that:
  - streams the shuffled FineWeb-Edu source,
  - counts tokens with the LLaMA-2 tokenizer,
  - writes a deterministic JSONL prefix,
  - converts the result into Megatron binary format.
- Added and validated short quick-run sbatches for end-to-end runtime checks before longer runs.
- Updated the 350M Transformer++ AdamW launcher to use FineWeb-Edu and the LLaMA-2 tokenizer.
- Added a 350M Gated DeltaNet launcher under the same data, tokenizer, and optimizer setup, including the Flash Linear Attention runtime dependencies needed by the Megatron GDN path.
- Added a basic 350M plain DeltaNet launcher and module path as an ungated ablation under the same overall training recipe.
- Updated the Clariden / Alps launch path and dependency installation so the jobs run correctly in the containerized environment.

#### Transformer++ softmax

- launcher:
  [_research/launch/transformer-pp-350m-adamw.sbatch](../_research/launch/transformer-pp-350m-adamw.sbatch)
- no new attention module was added
- the work here was mostly configuration and runtime bring-up:
  - switch to FineWeb-Edu,
  - switch to LLaMA-2 tokenizer,
  - add Clariden quick-run / near-full launch settings,
  - keep AdamW fixed
- architecture stays on the standard Megatron / Transformer++ path with:
  - softmax self-attention,
  - Transformer Engine kernels,
  - RoPE,
  - grouped-query attention,
  - RMSNorm and SwiGLU

#### Gated DeltaNet

- module:
  [gated_delta_net.py](../megatron/core/ssm/gated_delta_net.py)
- launcher:
  [_research/launch/transformer-pp-350m-gdn.sbatch](../_research/launch/transformer-pp-350m-gdn.sbatch)
- Clariden work:
  - enable Flash Linear Attention dependencies in
    [install_python_deps.sh](../_research/launch/install_python_deps.sh)
  - add matching quick-run / near-full launchers
  - keep dataset, tokenizer, optimizer, batch size, and sequence length aligned with softmax
  - calibrate from 20 to 22 layers to get closer in size to the softmax run
- in this wrapper, GDN includes:
  - `--experimental-attention-variant gated_delta_net`
  - linear-attention layers inserted via `--linear-attention-freq 3`
  - short convolution preprocessing on `q`, `k`, and `v`
  - a learned delta-rule decay gate
  - a separate output gate after normalization

#### Plain DeltaNet

- module:
  [delta_net.py](../megatron/core/ssm/delta_net.py)
- wiring:
  [experimental_attention_variant_module_specs.py](../megatron/core/models/gpt/experimental_attention_variant_module_specs.py)
  and
  [transformer_config.py](../megatron/core/transformer/transformer_config.py)
- throughput accounting:
  [training.py](../megatron/training/training.py)
- launchers:
  [_research/launch/transformer-pp-350m-deltanet.sbatch](../_research/launch/transformer-pp-350m-deltanet.sbatch)
  and
  [_research/launch/transformer-pp-350m-deltanet-smoke.sbatch](../_research/launch/transformer-pp-350m-deltanet-smoke.sbatch)
- relative to GDN, plain DeltaNet:
  - uses `--experimental-attention-variant delta_net`
  - calls the plain Flash Linear Attention delta-rule kernel
  - removes the learned decay-gate path
  - removes the separate output gate after normalization
- I kept the surrounding training recipe aligned with GDN:
  - same FineWeb-Edu 50M dataset,
  - same LLaMA-2 tokenizer,
  - same AdamW optimizer,
  - same micro-batch size, global batch size, and sequence length,
  - same hybrid insertion frequency for linear-attention layers
- I also kept the short convolution and q/k normalization structure; because
  the plain DeltaNet wrapper uses the same head count for q/k and v, I adjusted
  the head layout to preserve the same overall q/k and value widths used in the
  GDN run

---
---

## Week 2

### Week 1 Recap

While revisiting the finished Clariden logs, I noticed that I had initially
mixed up the throughput units. The logs report both `tokens/s/GPU` and
`TFLOP/s/GPU`, and for runtime estimation the relevant quantity is
`tokens/s/GPU`, not `TFLOP/s/GPU`. More precisely:

- `tokens/s/GPU` comes from the local Apertus logging hook in
  [_research/logging_patch/hooks.py](../_research/logging_patch/hooks.py),
  where it is computed from wall-clock deltas between successive training-log
  calls;
- `TFLOP/s/GPU` comes from Megatron's own `--log-throughput` path in
  [megatron/training/training.py](../megatron/training/training.py).

The exact Week 1 recap from the final training-step log lines is:

| Baseline | Params | Final train loss | Final val loss | Val PPL | Throughput (ktokens/s/GPU) | Throughput (TFLOP/s/GPU) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer++ softmax (24L) | 317.2M | 5.9744 | 6.0082 | 406.8 | 143.3 | 331.0 |
| Gated DeltaNet (22L) | 313.4M | 5.1020 | 5.1217 | 167.6 | 161.4 | 302.0 |
| DeltaNet (24L) | 319.5M | 6.2663 | 6.3069 | 548.3 | 152.1 | 294.3 |

These values were read directly from the finished Clariden logs:

- `transformer-pp-350m-adamw-smoke-1929006.log`
- `transformer-pp-350m-gdn-smoke-1929007.log`
- `transformer-pp-350m-deltanet-smoke-1929951.log`

### Runtime Planning

I first tried scheduling the longer runs by reserving 12 hours for the
conversion step and 10 hours for each training step. In practice, the first
conversion job kept being postponed in the `normal` partition, so I estimated
the expected runtimes from the Week 1 quick-run measurements before
resubmitting.

The runtime planning reuses the same training setup as the 350M Transformer++
baseline launcher
[_research/launch/transformer-pp-350m-adamw.sbatch](../_research/launch/transformer-pp-350m-adamw.sbatch),
unless explicitly overridden in the Gated DeltaNet and DeltaNet launchers. In
particular, this means:

- micro-batch size = 16
- global batch size = 128
- sequence length = 4096
- tokens per iteration = `128 * 4096 = 524,288`
- data-parallel training on 4 GPUs with `TP=1`, `PP=1`, `CP=1`
- AdamW with `lr = 3e-4`, `min-lr = 3e-5`, WSD schedule, `bf16`
- FineWeb-Edu data and the LLaMA-2 SentencePiece tokenizer

This gives:

- `1B / 524,288 = 1907` iterations
- `15B / 524,288 = 28,610` iterations

Using the measured `tokens/s/GPU` values from the recap table above, the
approximate per-step times are:

- Transformer++ softmax: `~0.91 s/iter`
- Gated DeltaNet: `~0.81 s/iter`
- DeltaNet: `~0.86 s/iter`

This implies the following training-only runtimes:

| Baseline | 1B tokens | 15B tokens |
| --- | ---: | ---: |
| Transformer++ softmax | ~29.1 min | ~7.3 h |
| Gated DeltaNet | ~25.8 min | ~6.5 h |
| DeltaNet | ~27.4 min | ~6.9 h |

From these estimates, the practical conclusion is:

- `1B` conversion should be resubmitted separately with a shorter walltime
  target (`6h`) to improve schedulability;
- `1B` training jobs should fit comfortably within `2h`;
- `15B` training jobs should be kept around `8h`;
- the `15B` conversion should be launched only after the `1B` stage is
  complete, instead of queueing the entire `1B -> 15B` chain at once.

One operational detail that became clear during resubmission is that the
FineWeb-Edu dataset is not stored locally on Clariden as a ready-made training
prefix. The conversion job streams the raw FineWeb-Edu source from Hugging Face,
counts tokens with the LLaMA-2 tokenizer, writes a deterministic local JSONL
prefix, and only then converts that local prefix into Megatron `.bin` / `.idx`
files. In practice, the existence check is therefore not testing whether the
raw dataset exists on the cluster, but whether the converted local Megatron
prefix already exists at the expected scratch path.
