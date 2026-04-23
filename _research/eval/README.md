# CLER Evaluation

The first evaluation path is intentionally lightweight: run `lm-eval` on an
HF-compatible checkpoint for WikiText perplexity and LAMBADA. This keeps
training bring-up independent from Megatron-to-HF checkpoint conversion.

```bash
_research/eval/run_lm_eval_hf.sh /path/to/hf-or-fla-checkpoint
```

Defaults:

- tasks: `wikitext,lambada_openai`
- model backend: `hf`
- output: `_research/results/eval/<checkpoint-name>/`

Once checkpoint conversion is stable, expand the task list to the commonsense
suite used by DeltaNet-family papers: PIQA, HellaSwag, WinoGrande, ARC-Easy,
ARC-Challenge, SIQA, and BoolQ.
