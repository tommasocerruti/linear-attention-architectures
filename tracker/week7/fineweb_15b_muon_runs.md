# FineWeb-Edu 15B Muon Runs

This note is the handoff for running the three 350M Muon models on the
15B-token FineWeb-Edu LLaMA-2 dataset:

- Transformer++
- Gated DeltaNet
- DeltaNet

## Data

I searched the visible cluster scratch/store areas for a drop-in 15B
FineWeb-Edu Megatron dataset compatible with the earlier 1B CLER runs:
LLaMA-2 tokenizer, native Megatron indexed dataset, and one prefix usable as
`MEGATRON_DATA_PATH`. I did not find one.

What is present:

- the CLER/course checkouts have the 50M smoke set and the 1B LLaMA-2
  FineWeb-Edu set, but no 15B LLaMA-2 set;
- `/iopsstor/scratch/cscs/jpcoles/a06/swissai-fineweb-edu-score-2-filterrobots-merge`
  has large Megatron-style FineWeb-Edu shards, but this is a SwissAI/Apertus
  family dataset rather than the LLaMA-2 FineWeb-Edu prefix used by our 1B runs;
- other FineWeb locations under `bmessmer`, `asolergi`, `schlag`, and related
  paths are raw/link trees, Apertus-tokenized datasets, or access-limited
  directories rather than a verified LLaMA-2 CLER prefix.

The expected 15B prefix is:

```bash
/iopsstor/scratch/cscs/course_00206/cler/_research/results/data/fineweb_edu/fineweb_edu_15b_llama2/fineweb_edu_15b_llama2_text_document
```

Build it before launching training:

```bash
cd /iopsstor/scratch/cscs/course_00206/cler
export LLAMA2_TOKENIZER_MODEL=/iopsstor/scratch/cscs/course_00206/llama2-tokenizer/tokenizer.model
sbatch _research/data/convert_fineweb_edu.sbatch
```

If a teammate later finds or creates an equivalent shared prefix, pass it
without editing the scripts:

```bash
FINEWEB_15B_DATA_PREFIX=/absolute/prefix/without/.bin/or/.idx \
  sbatch _research/launch/transformer-pp-350m-fineweb-muon-15b.sbatch
```

After conversion, check that both files exist:

```bash
ls -lh _research/results/data/fineweb_edu/fineweb_edu_15b_llama2/fineweb_edu_15b_llama2_text_document.{bin,idx}
```

## Launch

From the repo root:

```bash
cd /iopsstor/scratch/cscs/course_00206/cler
sbatch _research/launch/transformer-pp-350m-fineweb-muon-15b.sbatch
sbatch _research/launch/transformer-pp-350m-gdn-muon-15b.sbatch
sbatch _research/launch/transformer-pp-350m-deltanet-muon-15b.sbatch
```

Each wrapper sets:

```bash
TRAIN_SAMPLES=3662109
EVAL_INTERVAL=1907
EVAL_ITERS=10
SAVE_INTERVAL=1907
CKPT_FORMAT=torch
NO_SAVE_OPTIM=1
NO_SAVE_RNG=1
```

`TRAIN_SAMPLES=3662109` is about 15B tokens at sequence length 4096.
`SAVE_INTERVAL=1907` writes a validation checkpoint about every 1B tokens,
matching the cadence used for the earlier 1B Muon checkpoint runs.

Checkpoint directories are created under:

```bash
_research/results/checkpoints/<model>-15b-ckpt-<STAMP>
```

By default these are model-only validation checkpoints, using
`--no-save-optim --no-save-rng`. If you need restartable training checkpoints,
override both flags when submitting:

```bash
NO_SAVE_OPTIM=0 NO_SAVE_RNG=0 sbatch _research/launch/transformer-pp-350m-fineweb-muon-15b.sbatch
```

## Monitoring

Logs go to:

```bash
_research/results/runs/%x-%j.log
```

Useful commands:

```bash
squeue -u "$USER"
tail -f _research/results/runs/transformer-pp-350m-fineweb-muon-15b-<jobid>.log
tail -f _research/results/runs/transformer-pp-350m-gdn-muon-15b-<jobid>.log
tail -f _research/results/runs/transformer-pp-350m-deltanet-muon-15b-<jobid>.log
```

## Overrides

Common safe overrides:

```bash
STAMP=20260516-15b sbatch _research/launch/transformer-pp-350m-gdn-muon-15b.sbatch
SAVE_INTERVAL=500 sbatch _research/launch/transformer-pp-350m-deltanet-muon-15b.sbatch
WANDB_API_KEY=<key> sbatch _research/launch/transformer-pp-350m-fineweb-muon-15b.sbatch
```

Avoid setting `TRAIN_ITERS` unless you intentionally want a shorter run; it
overrides the 15B `TRAIN_SAMPLES` target.

## After Training

Validate checkpoints with the native Megatron `lm_eval` path, not HF eval,
unless the checkpoint has been explicitly converted. Reuse the static local
server setup from `tracker/lm_eval_native_megatron_guide.md`.
