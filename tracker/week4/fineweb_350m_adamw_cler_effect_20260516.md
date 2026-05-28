# Week 4: 350M FineWeb-Edu/LLaMA2 AdamW CLER Effect

All runs used FineWeb-Edu tokenized with LLaMA2, `TRAIN_SAMPLES=244140`, final
iteration `1907`, sequence length `4096`, global batch size `128`, microbatch
`2`, 1 Clariden GH200 node with 4 GPUs, bf16, AdamW, WSD/minus-sqrt schedule,
and torch.compile for the PyTorch linear rule.

The CLER runs used scalar receiver gamma, raw routed residuals, gamma init
`0.0`, and the corrected carry-through across SDPA layers. They did not use
per-head gamma.

| Run | Job | W&B | Elapsed | Avg sec/iter | Avg tokens/s/GPU | Avg TFLOP/s/GPU | Final train loss | Final val loss / PPL | Best val loss / PPL |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| DeltaNet PyTorch AdamW | 2236098 | [run](https://wandb.ai/cler/clerv1-runs/runs/23xcd0t4) | 06:32:23 | 12.331 | 11174 | 18.30 | 3.432178 | 3.406819 / 30.1691 | 3.403113 / 30.0575 |
| CLER-DeltaNet AdamW | 2236099 | [run](https://wandb.ai/cler/clerv1-runs/runs/5vau5oe5) | 06:44:42 | 12.707 | 10849 | 17.77 | 3.451447 | 3.425838 / 30.7484 | 3.424568 / 30.7094 |
| GDN PyTorch AdamW | 2236100 | [run](https://wandb.ai/cler/clerv1-runs/runs/de1zr2it) | 07:55:55 | 14.937 | 8980 | 15.44 | 3.269968 | 3.255381 / 25.9295 | 3.251918 / 25.8399 |
| CLER-Gated AdamW | 2236101 | [run](https://wandb.ai/cler/clerv1-runs/runs/ys8xn44f) | 08:27:32 | 15.927 | 8407 | 14.46 | 3.271358 | 3.256151 / 25.9495 | 3.252717 / 25.8605 |

## Matched Comparisons

| Comparison | Final val delta | Best val delta | Throughput delta |
|---|---:|---:|---:|
| CLER-DeltaNet minus DeltaNet | +0.019019 | +0.021455 | -324 tokens/s/GPU |
| CLER-Gated minus GDN | +0.000770 | +0.000799 | -573 tokens/s/GPU |
| GDN minus DeltaNet | -0.151438 | -0.151195 | -2194 tokens/s/GPU |

Positive validation deltas mean CLER is worse.

## Validation Trajectory

| Iteration | DeltaNet AdamW | CLER-DN AdamW | GDN AdamW | CLER-Gated AdamW |
|---:|---:|---:|---:|---:|
| 100 | 6.503905 | 6.490861 | 5.357415 | 5.358899 |
| 200 | 5.215848 | 5.204655 | 4.523809 | 4.528111 |
| 500 | 4.281820 | 4.283229 | 3.910759 | 3.911216 |
| 1000 | 3.784249 | 3.806400 | 3.519524 | 3.520655 |
| 1500 | 3.512481 | 3.540632 | 3.329937 | 3.330700 |
| 1900 | 3.403113 | 3.424568 | 3.251918 | 3.252717 |
| 1907 | 3.406819 | 3.425838 | 3.255381 | 3.256151 |

## CLER Sidecars

| Run | Gamma abs-mean | Gamma max-abs | Residual abs-mean mean | Residual max-abs |
|---|---:|---:|---:|---:|
| CLER-DeltaNet AdamW | 0.00269 | 0.00629 | 0.02217 | 4.4375 |
| CLER-Gated AdamW | 0.00309 | 0.00702 | 0.04632 | 2.5625 |

Sidecar paths:

- `/users/course_00252/cler/_research/results/performance/cler-deltanet-v1-carry-350m-llama2-fwe1b-adamw-compile10h-2236099.cler_gamma.jsonl`
- `/users/course_00252/cler/_research/results/performance/cler-deltanet-v1-carry-350m-llama2-fwe1b-adamw-compile10h-2236099.cler_residual.jsonl`
- `/users/course_00252/cler/_research/results/performance/cler-gated-v1-carry-350m-llama2-fwe1b-adamw-compile10h-2236101.cler_gamma.jsonl`
- `/users/course_00252/cler/_research/results/performance/cler-gated-v1-carry-350m-llama2-fwe1b-adamw-compile10h-2236101.cler_residual.jsonl`

## Conclusion

AdamW does not reveal a useful scalar-CLER effect. CLER-DeltaNet is materially
worse than DeltaNet, and CLER-Gated is slightly worse than GDN.

The important architecture result is that GDN is much stronger than DeltaNet
under AdamW. This supports the idea that Muon was compressing the observed
DeltaNet/GDN gap. However, changing optimizer did not make scalar CLER useful.

Next useful run: CLER-Gated AdamW with per-head gamma and gamma init `1e-2`,
raw residual, compared against the already completed GDN AdamW baseline
`2236100`. This directly tests the best CLER-Gated Muon ablation under the
optimizer where GDN separates clearly from DeltaNet. Run a debug smoke first.

## Follow-up Scheduled 2026-05-17

I did not schedule another scalar-gamma initialization sweep. Existing scalar
AdamW and Muon evidence shows that scalar CLER is either neutral or worse, and
the `1e-2`/`5e-2` initialization runs did not create a meaningful separation.

The one useful missing AdamW ablation is CLER-Gated with per-head receiver
gamma, raw residual, gamma init `1e-2`, and torch.compile. This isolates the
only CLER change that helped at all under Muon, while comparing against the
completed AdamW GDN baseline `2236100`.

| Run | Job | Debug gate | Wrapper | Status |
|---|---:|---:|---|---|
| CLER-Gated per-head AdamW | 2282970 | 2282956 passed 8/8 | `_research/launch/transformer-pp-350m-fwe1b-cler-gated-v1-headgamma-gamma1e-2-adamw-9h58-compile.sbatch` | reached final validation; see 2026-05-19 result below |

Debug smoke `2282956` completed in `00:03:52`, wrote W&B to
`cler/clerv1-runs`, reached 8/8 iterations, and produced both sidecars:

- `_research/results/performance/debug-cler-gated-v1-headgamma-350m-llama2-fwe50m-adamw-gamma1e-2-compile-2282956.cler_gamma.jsonl`
- `_research/results/performance/debug-cler-gated-v1-headgamma-350m-llama2-fwe50m-adamw-gamma1e-2-compile-2282956.cler_residual.jsonl`

Smoke sanity checks: 8 gamma rows, 8 residual rows, 112 per-head gamma values
per row, and 14 CLER-capable GDN layers.

## Follow-up Scheduled 2026-05-18

The per-head AdamW run `2282970` is still running. At iteration 900 its
validation loss is worse than both matched references:

| Iteration | GDN AdamW `2236100` | Scalar CLER-Gated AdamW `2236101` | Per-head CLER-Gated AdamW `2282970` |
|---:|---:|---:|---:|
| 100 | 5.357415 | 5.358899 | 5.351056 |
| 200 | 4.523809 | 4.528111 | 4.517053 |
| 300 | 4.207028 | 4.210221 | 4.208296 |
| 500 | 3.910759 | 3.911216 | 3.912466 |
| 700 | 3.735571 | 3.736762 | 3.739988 |
| 900 | 3.596911 | 3.598134 | 3.602329 |
| 1000 | 3.519524 | 3.520655 | 3.525148 |

Interpretation so far: per-head gamma briefly improves early validation, but
does not appear to converge faster by mid-run.

Last scheduled capacity attempt: per-channel receiver gamma on CLER-Gated
AdamW. This means one gamma for each local value element `[value_head,
value_channel]`: `14 * 8 * 64 = 7168` CLER gamma values at 350M.

| Run | Job | Debug gate | Wrapper | Status |
|---|---:|---:|---|---|
| CLER-Gated per-channel AdamW | 2291777 | 2291733 passed 8/8 | `_research/launch/transformer-pp-350m-fwe1b-cler-gated-v1-channelgamma-gamma1e-2-adamw-9h58-compile.sbatch` | completed 0:0; see 2026-05-21 result below |

Debug smoke `2291733` completed in `00:03:51`, wrote W&B to
`cler/clerv1-runs`, reached 8/8 iterations, and produced 8 gamma/residual
sidecar rows. The final debug gamma row had `count=7168`,
`abs_mean=0.0100028`, and `max_abs=0.0108032`.

## Follow-up Result 2026-05-19: Per-Head CLER-Gated AdamW

Job `2282970` reached all `1907` iterations and wrote final validation to W&B,
but Slurm marked the batch job `FAILED` with exit `2:0` because the wrapper
reported a shell syntax error after W&B sync. The training step itself
completed and the run data are usable.

W&B: [run](https://wandb.ai/cler/clerv1-runs/runs/7wv1rlze)

| Run | Job | Final val loss / PPL | Best val loss / PPL | Avg tokens/s/GPU | Avg TFLOP/s/GPU | Sidecar gamma count |
|---|---:|---:|---:|---:|---:|---:|
| GDN PyTorch AdamW | 2236100 | 3.255381 / 25.9295 | 3.251918 / 25.8399 | 9051 | 15.23 | n/a |
| Scalar CLER-Gated AdamW | 2236101 | 3.256151 / 25.9495 | 3.252717 / 25.8605 | 8472 | 14.27 | 14 |
| Per-head CLER-Gated AdamW | 2282970 | 3.258292 / 26.0051 | 3.254543 / 25.9078 | 8910 | 14.98 | 112 |

Matched deltas:

| Comparison | Final val delta | Best val delta | Interpretation |
|---|---:|---:|---|
| Per-head CLER minus GDN | +0.002911 | +0.002625 | Worse than baseline |
| Per-head CLER minus scalar CLER | +0.002141 | +0.001826 | Worse than scalar CLER |

The early advantage at iterations 100-200 did not persist. By the end of
training, per-head CLER is worse than both matched GDN AdamW and scalar
CLER-Gated AdamW.

Final sidecar summaries at step `1900`:

| Sidecar | Count | Mean / abs-mean | Max abs |
|---|---:|---:|---:|
| `cler_gamma` | 112 | mean 0.009856, abs-mean 0.010046 | 0.026245 |
| `cler_residual` | 14 layers | abs-mean mean 0.046136 | 2.28125 |

Conclusion: per-head gamma should no longer be described as the best completed
AdamW CLER setting. Among completed AdamW CLER-Gated runs, scalar CLER remains
the least-worse result, but it is still worse than GDN. The per-channel result
below confirms that static channel capacity does not rescue CLER-Gated AdamW.

## Follow-up Result 2026-05-21: Per-Channel CLER-Gated AdamW

Job `2291777` completed successfully with exit `0:0` in `08:11:31`.

W&B: [run](https://wandb.ai/cler/clerv1-runs/runs/lk38ctsx)

| Run | Job | Final val loss / PPL | Best val loss / PPL | Avg tokens/s/GPU | Avg TFLOP/s/GPU | Sidecar gamma count |
|---|---:|---:|---:|---:|---:|---:|
| GDN PyTorch AdamW | 2236100 | 3.255381 / 25.9295 | 3.251918 / 25.8399 | 9047 | 15.23 | n/a |
| Scalar CLER-Gated AdamW | 2236101 | 3.256151 / 25.9495 | 3.252717 / 25.8605 | 8469 | 14.27 | 14 |
| Per-head CLER-Gated AdamW | 2282970 | 3.258292 / 26.0051 | 3.254543 / 25.9078 | 8911 | 14.98 | 112 |
| Per-channel CLER-Gated AdamW | 2291777 | 3.257861 / 25.9939 | 3.253723 / 25.8865 | 8750 | 14.75 | 7168 |

Matched deltas:

| Comparison | Final val delta | Best val delta | Interpretation |
|---|---:|---:|---|
| Per-channel CLER minus GDN | +0.002480 | +0.001805 | Worse than baseline |
| Per-channel CLER minus scalar CLER | +0.001710 | +0.001006 | Worse than scalar CLER |
| Per-channel CLER minus per-head CLER | -0.000431 | -0.000820 | Better than per-head, but both are worse than scalar/GDN |

Validation trajectory for GDN-family AdamW runs:

| Iteration | GDN | Scalar CLER | Per-head CLER | Per-channel CLER |
|---:|---:|---:|---:|---:|
| 100 | 5.357415 | 5.358899 | 5.351056 | 5.359388 |
| 200 | 4.523809 | 4.528111 | 4.517053 | 4.511030 |
| 500 | 3.910759 | 3.911216 | 3.912466 | 3.912965 |
| 1000 | 3.519524 | 3.520655 | 3.525148 | 3.523217 |
| 1500 | 3.329937 | 3.330700 | 3.333242 | 3.332601 |
| 1900 | 3.251918 | 3.252717 | 3.254543 | 3.253723 |
| 1907 | 3.255381 | 3.256151 | 3.258292 | 3.257861 |

Final sidecar summaries at step `1900`:

| Run | Gamma abs-mean | Gamma max-abs | Residual abs-mean mean | Residual max-abs |
|---|---:|---:|---:|---:|
| Scalar CLER-Gated AdamW | 0.00309 | 0.00702 | 0.04632 | 2.5625 |
| Per-head CLER-Gated AdamW | 0.01005 | 0.02625 | 0.04614 | 2.28125 |
| Per-channel CLER-Gated AdamW | 0.01028 | 0.04297 | 0.04610 | 2.34375 |

Conclusion: per-channel gamma increases capacity and learns larger/more varied
weights, but it does not improve the training result. It is better than
per-head AdamW, worse than scalar AdamW, and worse than the matched GDN
baseline. Among completed AdamW CLER-Gated settings, scalar CLER remains the
least-worse configuration; none should be presented as an improvement over GDN.

## Follow-up Scheduled 2026-05-22: Pure Linear GDN vs CLER

The remaining architecture-pattern loophole is that all completed main 350M
runs used the hybrid `linear_attention_freq=3` stack:

```text
GDN, GDN, SDPA, GDN, GDN, SDPA, ...
```

The pure-linear follow-up uses `LINEAR_ATTENTION_FREQ=21`, which gives all 20
layers as GDN mixers for this 20-layer model. This is deliberately a matched
two-run pair: GDN baseline vs scalar CLER-Gated, both AdamW, torch.compile,
350M / 1B FineWeb-Edu/LLaMA2.

Debug smokes passed first:

| Run | Debug job | Result | W&B |
|---|---:|---|---|
| pure scalar CLER-Gated | 2342520 | 8/8 iterations, exit 0:0 | `psdb6tcs` |
| pure GDN | 2342521 | 8/8 iterations, exit 0:0 | `42fvsihv` |

Full jobs queued:

| Run | Job | Wrapper |
|---|---:|---|
| pure GDN PyTorch AdamW | 2342564 | `_research/launch/transformer-pp-350m-fwe1b-pure-gdn-pytorch-adamw-11h58-compile.sbatch` |
| pure scalar CLER-Gated AdamW | 2342565 | `_research/launch/transformer-pp-350m-fwe1b-pure-cler-gated-v1-scalar-adamw-11h58-compile.sbatch` |

Interpretation target: compare `2342565` only against `2342564`. Do not mix
this with the hybrid GDN baseline unless explicitly discussing the effect of
removing SDPA layers.

## Follow-up Result 2026-05-24: CLER-v2 Output Injection

This follow-up changed only the CLER injection site: the routed residual is
added to the receiver readout before output normalization/projection instead of
only perturbing the value target. The debug smoke passed, so the wiring is
correct; the full run then tells us whether the new site actually helps.

| Run | Job | Result | Final train loss | Best train loss | Median tokens/s/GPU (last 100) | Duration | Sidecar gamma abs-mean | Sidecar gamma max-abs | Residual abs-mean mean | Residual max-abs |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Debug CLER-v2 output | 2368645 | 10/10 iterations, exit 0:0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Full CLER-v2 output | 2368738 | 1907 iterations, exit 0:0 | 3.270266 | 3.166018 | 8807.31 | 8.14 h | 0.000475 | 0.000923 | 0.04919 | 2.8594 |

Relative to the earlier scalar CLER-Gated AdamW run (`2236101`), output injection
is a small train-loss and throughput improvement, but it still does not beat the
matched GDN baseline (`2236100`) on training loss. The learned gamma is also
much smaller than before, which suggests that the model continues to suppress
the routed correction rather than exploiting it more aggressively.

Conclusion: moving the CLER correction to the receiver readout is a cleaner
test of the "too indirect" hypothesis, but it still does not produce a useful
CLER gain at this scale.

## Follow-up Result 2026-05-27: Pure Linear GDN vs Scalar CLER-Gated

Both jobs completed successfully. This pair did not log validation rows, so it
should be treated as a training-curve and throughput check rather than a final
validation comparison.

| Run | Job | Final train loss | Best train loss | Median tokens/s/GPU (last 100) | Duration | Sidecar gamma abs-mean | Sidecar gamma max-abs | Residual abs-mean mean | Residual max-abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pure GDN PyTorch AdamW | 2342564 | 3.448133 | 3.338206 | 6510.41 | 10.79 h | n/a | n/a | n/a | n/a |
| pure scalar CLER-Gated AdamW | 2342565 | 3.450032 | 3.339784 | 6064.50 | 11.56 h | 0.00256 | 0.00729 | 0.04919 | 2.8594 |

Matched deltas:

| Comparison | Final train delta | Best train delta | Throughput delta |
|---|---:|---:|---:|
| CLER minus GDN | +0.001899 | +0.001578 | -446 tokens/s/GPU |

Conclusion: removing the SDPA layers does not rescue scalar CLER. The pure
GDN baseline is still slightly better in training loss and faster in
throughput, while the CLER sidecars remain nonzero but small enough that the
injected correction is still weak relative to the main value stream.
