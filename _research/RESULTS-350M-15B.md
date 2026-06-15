# 350M / 15B-token final runs — results + checkpoints (for downstream evals)

Updated: 2026-06-12. Contact: Lingfeng. All runs: 350M-class (20L / 1024H / 2816 FFN), hybrid
linear/SDPA stack (`linear_attention_freq=3`), seq 4096, GBS 128, bf16, Muon (adaptive_muon/normuon,
LR 3.6e-4, WSD), LLaMA-2 tokenizer, seed 42, 15B FineWeb-Edu tokens (token-count slice of
`HuggingFaceFW/fineweb_edu_100BT-shuffled`, read from the 62B `_tc` Megatron prefix),
TRAIN_SAMPLES=3662109 → 28,610 iterations. W&B: `cler/clerv1-runs`, group `350M-15B-FINAL-20260610`
(one W&B run per experiment, run name = checkpoint dir name).

## Final validation loss (last eval @ iter 28610 = 15B tokens)

| run | final val loss | Δ vs own baseline |
|---|---:|---:|
| GDN-350M-15B-MUON | 2.341689 | — |
| CLER-V-350M-15B-MUON | 2.335839 | −0.0059 |
| CLER-H-350M-15B-MUON | 2.337488 | −0.0042 |
| GDN-ATTNRES-350M-15B-MUON | 2.344661 | +0.0030 (worse) |
| DELTANET-350M-15B-MUON | 2.334680 | — |
| **DELTANET-CLER-V-350M-15B-MUON** | **2.333069** | −0.0016 (best overall) |
| DELTANET-CLER-H-350M-15B-MUON | 2.334493 | −0.0002 |
| DELTANET-ATTNRES-350M-15B-MUON | (still training) | |

Mechanisms: CLER-V/H = per-GDN/DeltaNet-layer zero-init projection of the mixer's internal write
value v (V) or write residual r = v − Sφ(k) (H) added to the residual stream (`--cler-enabled
--cler-hidden-routing [--cler-hidden-route-value]`, full-rank). ATTNRES = Full Attention Residuals
(softmax over depth on sub-layer outputs, `--attn-res-enabled`).

## Checkpoints (Megatron `torch` format, final iteration only, params + Muon optimizer state)

Base dir: `/iopsstor/scratch/cscs/lingfeng/cler/_research/results/checkpoints/`
(iopsstor Lustre scratch; group `infra01` readable; ~3–4 GB each)

```
/iopsstor/scratch/cscs/lingfeng/cler/_research/results/checkpoints/GDN-350M-15B-MUON/iter_0028610
/iopsstor/scratch/cscs/lingfeng/cler/_research/results/checkpoints/CLER-H-350M-15B-MUON/iter_0028610
/iopsstor/scratch/cscs/lingfeng/cler/_research/results/checkpoints/CLER-V-350M-15B-MUON/iter_0028610
/iopsstor/scratch/cscs/lingfeng/cler/_research/results/checkpoints/GDN-ATTNRES-350M-15B-MUON/iter_0028610
/iopsstor/scratch/cscs/lingfeng/cler/_research/results/checkpoints/DELTANET-350M-15B-MUON/iter_0028610
/iopsstor/scratch/cscs/lingfeng/cler/_research/results/checkpoints/DELTANET-CLER-H-350M-15B-MUON/iter_0028610
/iopsstor/scratch/cscs/lingfeng/cler/_research/results/checkpoints/DELTANET-CLER-V-350M-15B-MUON/iter_0028610
(DELTANET-ATTNRES-350M-15B-MUON — will appear at the same pattern when training finishes)
```

## How to load

Pass the PARENT directory (without `/iter_0028610`) as `--load`; Megatron reads
`latest_checkpointed_iteration.txt`. Use this repo on the current release branch (the DELTANET-CLER-*
checkpoints need the delta_net.py CLER support committed alongside this file). Reproduce each
architecture's exact flags with the matching launcher in `_research/launch/final-<NAME>.sbatch`.
Tokenizer: Llama2Tokenizer, model file at
`/iopsstor/scratch/cscs/lingfeng/llama2-tokenizer/tokenizer.model`.

NOTE (scratch retention): iopsstor scratch is subject to cleanup policies — if evals are weeks away,
copy the checkpoints to your own storage (use the `xfer` partition for the transfer, not a login node).
