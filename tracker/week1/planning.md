# DeltaNet Variant Investigation Report

`n/r` = not reported in accessible paper/model-card text.

Sources: [DeltaNet (Schlag et al., 2021)](https://arxiv.org/abs/2102.11174), [Gated DeltaNet (Yang et al., 2024)](https://arxiv.org/abs/2412.06464), [GLA / Flash Linear Attention (Yang et al., 2024)](https://arxiv.org/abs/2312.06635), [RLA / RDN (Lai et al., 2025)](https://arxiv.org/abs/2509.25223), [Kimi Linear (Kimi Team, 2025)](https://arxiv.org/abs/2510.26692), [GLA 1.3B model card](https://huggingface.co/fla-hub/gla-1.3B-100B), [GLA 340M model card](https://huggingface.co/fla-hub/gla-340M-15B), [NVLabs GatedDeltaNet repo](https://github.com/NVlabs/GatedDeltaNet), [Kimi Linear repo](https://github.com/MoonshotAI/Kimi-Linear).

---
## Table 1a — Pretraining Datasets

| Dataset | Tokens | Used in | In repo? | Integration | Notes |
| --- | --- | --- | --- | --- | --- |
| SlimPajama | 15B / 100B subsets | GLA (15B for 340M, 100B for 1.3B); RLA/RDN (100B) | No | EASY | Standard Megatron data path fits it; recommended primary dataset for reproducibility since both GLA and RLA/RDN used it |
| FineWeb-Edu | 15B / 100B subsets | Gated DeltaNet (100B); GDN ablations (15B) | No | EASY | Used for all GDN experiments; Megatron has generic preprocessing support |
| WikiText-103 | ~103M words | DeltaNet (train + eval); RLA/RDN (eval) | Partly | EASY | Megatron examples/docs reference it; tokenizer/preprocess path exists |
| K2 Corpus | 1.4T / 5.7T | Kimi Linear | No | N/A | Proprietary Moonshot corpus; not publicly available |
| Nemotron ClimbMix (`climbmix_small`) | ~6–12B | Repo only (Gipfelsturm) | Yes | INCLUDED | Already converted to Megatron `.bin/.idx`; useful for quick smoke tests |
| WMT14 En-De | n/r in tokens | DeltaNet (MT experiments) | Partly | HARD | Repo has `preprocess_data_nmt.py`, but launcher is GPT-pretrain oriented |

## Table 1b — Evaluation Benchmarks

| Benchmark | Type | Used in | In repo? | Integration | Notes |
| --- | --- | --- | --- | --- | --- |
| WikiText-103 perplexity | LM perplexity | DeltaNet; GDN; GLA; RLA/RDN | Partly | EASY | Standard LM eval; cheap and fast |
| LAMBADA (LMB) | LM / zero-shot | GDN; RLA/RDN | No | EASY | Via `lm-eval` harness |
| Commonsense suite (PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, SIQA, BoolQ) | Zero-shot reasoning | GDN; RLA/RDN | No | MEDIUM | Requires `lm-eval` harness install + checkpoint export |
| MMLU | Few-shot knowledge | RLA/RDN; Kimi Linear | Partly | EASY | ModelOpt MMLU script exists using `cais/mmlu` from HF |
| MMLU-Pro | Few-shot knowledge | Kimi Linear | No | MEDIUM | Via `lm-eval` harness |
| Recall suite (SWDE, SQuAD, FDA, TriviaQA, NQ, DROP, NIAH) | In-context recall | GDN; RLA/RDN | No | HARD | Custom eval scripts needed; measures associative recall ability |
| LongBench / LongBench V2 | Long-context QA | GDN; Kimi Linear | No | HARD | 14 tasks covering single/multi-doc QA, summarization, few-shot, code |
| RULER | Long-context synthetic | GDN; Kimi Linear | No | HARD | Needle-in-a-haystack style; external NVIDIA/RULER tooling needed |
| S-NIAH | Long-context synthetic | GDN | No | HARD | Single-needle-in-a-haystack variant |
| HELMET-ICL | Long-context ICL | Kimi Linear | No | HARD | In-context learning benchmark |
| Synthetic tasks (Palindrome, MQAR, Stack) | Synthetic reasoning | Kimi Linear | No | MEDIUM | Tests state tracking and associative recall; small custom setups |
| GSM8k, HumanEval | Math / code | RLA/RDN (ablation) | No | MEDIUM | Standard math and code benchmarks |
| MATH 500, AIME 2025 | Math reasoning | Kimi Linear (RL) | No | HARD | Used for RL training evaluation |

---

## Table 2 — Pretraining Configurations from Literature

### DeltaNet

| Config | Params | Layers | Hidden | Heads | FFN        | Seq len | Tokens      | Dataset      |
| ------ | ------ | ------ | ------ | ----- | ---------- | ------- | ----------- | ------------ |
| Small  | 40M    | 16     | 128    | 8     | 2048 (16x) | n/r     | ~103M words | WikiText-103 |
| Medium | 90M    | 16     | 256    | n/r   | n/r        | n/r     | ~103M words | WikiText-103 |

### GLA (Flash Linear Attention)

From HuggingFace model cards (`config.json`):

| Config | Params | Layers | Hidden | Heads                                     | Seq len | Tokens | Dataset    |
| ------ | ------ | ------ | ------ | ----------------------------------------- | ------- | ------ | ---------- |
| 340M   | 340M   | 24     | 1024   | 4 (default but did different abeltations) | n/r     | 15B    | SlimPajama |
| 1.3B   | 1.3B   | 24     | 2048   | 4 (default but did different abeltations) | n/r     | 100B   | SlimPajama |

Activation: SwiSH. Output gate: yes. Gated key: yes. Gated value: no. Vocab: 32,000 (LLaMA-2 tokenizer).

### Gated DeltaNet

The paper uses a LLaMA-style macro architecture but does not publish an explicit layer/dimension table. Known details:

| Config         | Params | Tokens | Dataset     | Head dim                 | Seq len | Notes                                          |
| -------------- | ------ | ------ | ----------- | ------------------------ | ------- | ---------------------------------------------- |
| Pure recurrent | 400M   | 100B   | FineWeb-Edu | 128 (best from ablation) | 4096    | SwiGLU MLP, L2-norm on Q/K                     |
| Pure recurrent | 1.3B   | 100B   | FineWeb-Edu | 128                      | 4096    | Same setup                                     |
| Hybrid H1      | 1.3B   | 100B   | FineWeb-Edu | 128                      | 4096    | GDN + SWA (2K window)                          |
| Hybrid H2      | 1.3B   | 100B   | FineWeb-Edu | 128                      | 4096    | Mamba2 + GDN + SWA (2K window)                 |

### RLA / Residual DeltaNet

From paper Table 6 and Table 7:

| Config | Params | Layers | Hidden | Intermediate | Heads | GQA groups | Head dim | Softmax layers | Linear layers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer baseline | 1.51B | 16 | 2048 | 8192 | 32 | 4 | 128 | 16 | 0 |
| Linear attention (RLA/RDN/sGLA/GDN etc.) | 1.55B | 16 | 2048 | 8192 | 16 | 16 | 128 | 0 | 16 |

All trained on 100B tokens from SlimPajama. Tokenizer: n/r.

### Kimi Linear

All a bit more complex also due to MoE and diffferent releases etc.

| Config | Total params | Activated params | KDA:MLA ratio | Seq len (pretrain) | Seq len (final) | Tokens | Dataset |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fair comparison | 48B | 3B | 3:1 | 4096 | — | 1.4T | K2 corpus |
| Released checkpoint | 48B | 3B | 3:1 | 4096 | 1M | 5.7T | K2 corpus |

Scaling law experiment configs (from paper Table 2):

| Activated params | Heads | Layers | Hidden | Tokens | LR | Batch size |
| --- | --- | --- | --- | --- | --- | --- |
| 653M | 16 | 16 | 1216 | 38.8B | 2.006e-3 | 336 |
| 878M | 18 | 18 | 1376 | 59.8B | 1.790e-3 | 432 |
| 1.1B | 20 | 20 | 1536 | 85.2B | 1.617e-3 | 512 |
| 1.4B | 22 | 22 | 1632 | 102.5B | 1.486e-3 | 576 |
| 1.7B | 24 | 24 | 1776 | 128.0B | 1.371e-3 | 640 |

All scaling law models use context length 4096. KDA extends GDN by replacing the scalar forget gate with a channel-wise (per-dimension) forget gate vector.

---

## Table 3 — Optimizer and Training Hyperparameters

| Paper | Optimizer | Peak LR | Min LR | WD | Betas | Batch (tokens) | Warmup | Schedule | Grad clip | Init std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeltaNet | Adam | 2.5e-4 | n/r | n/r | default | seq batch 96 (small) / 56 (medium) | 2000 steps | n/r | n/r | n/r |
| GLA | AdamW | 3e-4 | 3e-5 | 0.01 | n/r | 0.5M (340M) / 2M (1.3B) | 0.5B / 1B tokens | Cosine | 1.0 | n/r |
| Gated DeltaNet | AdamW | 4e-4 | n/r | 0.1 | n/r | 0.5M | 1B tokens | Cosine | 1.0 | n/r |
| RLA / RDN | AdamW | 3e-4 | 3e-5 | 0.1 | n/r | 4M | 0.5B tokens | Cosine | 1.0 | 0.006 |
| Kimi Linear | MuonClip | 1.1e-3 | n/r | n/r | n/r | 32M | n/r | WSD | n/r | n/r |

Notes:
- DeltaNet also includes synthetic associative-retrieval tasks and WMT14 En-De MT; the row above uses the WikiText-103 LM setup.
- Gated DeltaNet uses the LLaMA-2 tokenizer (32K vocab).
- GLA uses initial and final LR of 3e-5.
- RLA/RDN initializes weights from a normal distribution with constant std 0.006.
- Kimi Linear uses the WSD (Warmup-Stable-Decay) schedule and the MuonClip optimizer (a variant of Muon with clipped updates). The Muon optimizer achieves ~2x computational efficiency vs AdamW at scale (demonstrated by Moonlight/Moonshot). All Kimi scaling law models share the same annealing and long-context activation phase from Kimi K2.
- Hardware: DeltaNet explicitly mentions 3x V100 (WMT14) and 2x V100 (WikiText LM). GDN throughput benchmarks use a single H100. Other papers underreport hardware.

---

## Evaluation Results Screenshots

### DeltaNet (2021)

Table 1 — WMT14 En-De BLEU scores:
![[Pasted image 20260422102849.png]]

Table 2 — WikiText-103 perplexity (small: 40M, medium: 90M):
![[Pasted image 20260422102914.png]]

Table 3 — WikiText-103 perplexity ablation (position encoding / attention normalization):
![[Pasted image 20260422102942.png]]

Table 4 — WikiText-103 perplexity without truncating context (vs Transformer-XL):
![[Pasted image 20260422102958.png]]

![[Pasted image 20260422102833.png]]

### Gated DeltaNet (2024)

Table 3 — Language modeling and zero-shot commonsense reasoning (1.3B, 100B tokens, FineWeb-Edu):
![[Pasted image 20260422103134.png]]

Table 4 — Recall-world retrieval tasks (input truncated to 2K tokens):
![[Pasted image 20260422103557.png]]

Figure 2 — Length extrapolation on six long-context benchmarks:
![[Pasted image 20260422103635.png]]

Table 5 — LongBench accuracy (14 tasks):
![[Pasted image 20260422103705.png]]

Figure 3 — Training throughput comparison (1.3B, single H100):
![[Pasted image 20260422103725.png]]

Table S.1 — GDN ablation study (400M, 15B tokens): macro design, normalization, feature maps, head dimensions:
![[Pasted image 20260422103808.png]]

Table S.2 — Hybrid ablation (500M, 15B tokens): GDN + SWA + Mamba2 layer orderings:
![[Pasted image 20260422103825.png]]

### GLA / Flash Linear Attention (2024)

![[Pasted image 20260422104318.png]]
![[Pasted image 20260422104408.png]]
![[Pasted image 20260422104447.png]]

### RLA / Residual DeltaNet (2025)

Table 2 — Language modeling, reasoning, commonsense (1.5B, 100B tokens, SlimPajama):
Table 3 — Recall-intensive benchmarks (DROP, FDA, NQ, SQD, SWDE, TQA, NIAH):
![[Pasted image 20260422105005.png]]

Table 4 — Ablation: residual fitting (RLA vs RLA w/o fitting, 50B tokens):
![[Pasted image 20260422105042.png]]

Table 5 — Ablation: gated output combination methods:
![[Pasted image 20260422105125.png]]

Table 6 — Model configuration; Table 7 — Training hyperparameters:
![[Pasted image 20260422105347.png]]

### Kimi Linear (2025)

Figure 1 — Performance vs decoding acceleration (MMLU-Pro, RULER) and TPOT vs decoding length:
![[Pasted image 20260422111434.png]]

Figure 4 — Synthetic tasks: palindrome, MQAR, stack (KDA vs GDN vs Mamba2):
![[Pasted image 20260422111504.png]]

Table 1 — Ablation: KDA-to-MLA hybrid ratio and component ablations:
![[Pasted image 20260422111614.png]]

Table 2 — Scaling law configs and fitted scaling curves (MLA vs Kimi Linear):
![[Pasted image 20260422111654.png]]

Evaluation benchmarks and configurations (Language, Math, Code, Chinese):
![[Pasted image 20260422111725.png]]

Table 3 — Pretrain performance (MLA vs GDN-H vs Kimi Linear at 1.4T tokens):
Table 4 — Post-SFT instruction-tuned performance:
![[Pasted image 20260422111829.png]]

Table 5 — Long-context benchmarks (RULER, MRCR, HELMET-ICL, LongBench V2, Frames, RepoQA, Long Code Arena):
![[Pasted image 20260422111851.png]]

Tables 8–9 — Kimi Linear Base and Instruct vs Moonlight across diverse tasks:
![[Pasted image 20260422111950.png]]

---

## Table 4 — Recommended Experiment Setup

| Component | Recommendation | Justification |
| --- | --- | --- |
| **Baseline architecture** | Hybrid Gated DeltaNet in Megatron | Only DeltaNet-family implementation already in the Gipfelsturm Megatron fork |
| **Target size** | 20 layers, hidden 1024, 16 heads, FFN 2816, GQA/KV heads 4 | Close to the repo's `350m` preset; preserves a known-good width/head regime |
| **FFN multiplier** | 2816 / 1024 = 2.75 | Matches the repo's current mid-scale SwiGLU style |
| **Attention mix** | `--experimental-attention-variant gated_delta_net` with `--linear-attention-freq 3` | Repo-tested path; yields a 2x GDN + 1x SDPA hybrid pattern |
| **Linear-attention block dims** | `linear-key-head-dim=64`, `linear-value-head-dim=64`, conv kernel `4` | Matches the shipped GDN functional test config |
| **Training dataset** | SlimPajama 15B subset (fast ablations) / 100B subset (full run) | Both GLA and RLA/RDN used SlimPajama; more reproducible and literature-comparable than ClimbMix or FineWeb-Edu; publicly available |
| **Tokenizer** | LLaMA-2 (32K vocab) for paper comparability; GPT-2 BPE if minimum friction needed | GDN/GLA papers used LLaMA-2; Gipfelsturm is currently wired for GPT-2 BPE |
| **Primary optimizer** | Muon | ~2x compute efficiency vs AdamW at scale (Moonlight); used by Kimi Linear (MuonClip variant); modern choice for linear attention research |
| **AdamW baseline** | Run one AdamW comparison to validate Muon matches | Confirm Muon reproduces AdamW loss curves on this architecture before committing to it for all ablations |
| **Learning rate (Muon)** | ~1e-3 peak (tune based on Muon scaling recommendations) | Muon typically uses higher LR than AdamW; Kimi Linear used 1.1e-3 |
| **Learning rate (AdamW)** | 3e-4 peak | Matches RLA/RDN and GLA baselines on SlimPajama |
| **Weight decay** | 0.1 | Matches GDN and RLA/RDN |
| **LR schedule** | Cosine decay (AdamW baseline) / WSD (Muon) | Cosine matches GDN/GLA/RLA literature; WSD is the Muon-native schedule |
| **Warmup** | 0.5B tokens | Matches RLA/RDN at 100B scale; GLA used 0.5B for 340M |
| **Min LR** | 3e-5 (AdamW) | Matches GLA and RLA/RDN |
| **Global batch** | 4M tokens | Matches RLA/RDN; at seq len 4096 this is ~976 sequences per optimizer step |
| **Sequence length** | 4096 | Matches GDN/Kimi training setup and Gipfelsturm launcher |
| **Gradient clip** | 1.0 | Universal across all papers |
| **Init std** | 0.006 | Matches RLA/RDN |
| **Precision** | bf16 | Already used in Megatron test configs and cluster setup |
| **Eval cadence** | WikiText-103 perplexity every checkpoint; LAMBADA accuracy; commonsense suite (PIQA, HellaSwag, WinoGrande, ARC) via `lm-eval`; recall benchmarks (SWDE, SQD, TQA, NQ, NIAH) if time allows | WikiText + LAMBADA are cheap; commonsense is the standard DeltaNet-family comparison; recall benchmarks test the core advantage of delta-rule attention |
| **MMLU** | After checkpoint export | Script exists in repo (ModelOpt) |
| **Fallback dataset** | If SlimPajama ingest is delayed, run on `climbmix_small` first | Validates kernels, scaling, and loss curves immediately with data already present |

---

## Integration TODO List

1. Install the external `flash-linear-attention` dependency in the Alps runtime; Megatron's GDN layer hard-fails without it.

2. Add a new `300m-gdn` launcher preset in `launch.sh` with `20/1024/2816/16` and the GDN-specific args from the functional test.

3. Add a SlimPajama download + conversion script mirroring the existing ClimbMix flow (`download_*` + `convert_data.sbatch`). Target a deterministic 15B-token subset for fast ablations and 100B for the full run.

4. Integrate the Muon optimizer into Megatron. Options: (a) use the open-source distributed Muon implementation from Moonlight, or (b) implement the Newton-Schulz orthogonalization step as a custom optimizer in Megatron's optimizer path. Run a single AdamW vs Muon comparison on the 15B subset to validate convergence.

5. Decide tokenizer path: stay on GPT-2 for fastest bring-up, or switch launcher/data prep to LLaMA-2 tokenizer for closer paper comparability.

6. Add a lightweight evaluation script path for WikiText-103 perplexity, LAMBADA, and wire the existing MMLU example as a post-checkpoint eval step. Install `lm-eval` harness for commonsense and recall benchmarks.

7. Only after baseline is stable: consider deeper work on pure DeltaNet, RDN/RLA, or Kimi-style KDA — these are all significant integrations, not small config flips.