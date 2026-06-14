# Gated DeltaNet 2

## GDN2 Runs

These jobs run the CLER checkout and import the external GDN2 kernels from
`/users/course_00206/GatedDeltaNet-2`. Nothing is vendored into CLER.

Set this up once before launching:

```bash
cd /users/course_00206/cler
source /users/course_00206/miniconda3/etc/profile.d/conda.sh
conda activate megatron-gdn2

# Important as tr1eder: otherwise the scripts default to /users/$USER.
export CODE_DIR=/users/course_00206/cler
export GDN2_REPO_DIR=/users/course_00206/GatedDeltaNet-2
export LLAMA2_TOKENIZER_MODEL=/iopsstor/scratch/cscs/course_00206/llama2-tokenizer/tokenizer.model

# Megatron dataset prefix, without .bin/.idx.
export MEGATRON_DATA_PATH=/iopsstor/scratch/cscs/tr1eder/cler/_research/results/data/fineweb_edu/fineweb_edu_1b_llama2_tc/fineweb_edu_1b_llama2_tc_text_document

test -f "$LLAMA2_TOKENIZER_MODEL"
test -f "$GDN2_REPO_DIR/lit_gpt/gdn2_ops/chunk_gdn2.py"
test -f "${MEGATRON_DATA_PATH}.bin" && test -f "${MEGATRON_DATA_PATH}.idx"
```

Run one script like this:

```bash
sbatch _research/launch/transformer-pp-350m-gdn2-fullattn-3to1.sbatch
```

Swap the script path for the variant you want.

### 350M Scripts

| Script | Optimizer | What it runs |
| --- | --- | --- |
| `_research/launch/transformer-pp-350m-gdn2.sbatch` | AdamW | 2:1 hybrid, `[GDN2, GDN2, standard attention]` repeated. This is the GDN2-paper-style hybrid slot; add/set window attention args if you need strict sliding-window attention. |
| `_research/launch/transformer-pp-350m-gdn2-fullattn-3to1.sbatch` | AdamW | 3:1 hybrid, `[GDN2, GDN2, GDN2, full attention]` repeated. This matches the other full-attention hybrid convention. |
| `_research/launch/transformer-pp-350m-gdn2-pure.sbatch` | AdamW | Pure GDN2, all 20 layers are GDN2. |
| `_research/launch/transformer-pp-350m-gdn2-muon.sbatch` | NorMuon | Same 2:1 hybrid as above, with `adaptive_muon + normuon`. |
| `_research/launch/transformer-pp-350m-gdn2-fullattn-3to1-muon.sbatch` | NorMuon | Same 3:1 full-attention hybrid, with Muon. |
| `_research/launch/transformer-pp-350m-gdn2-pure-muon.sbatch` | NorMuon | Pure GDN2, with Muon. |

The 350M full scripts default to about 15B training tokens:
`TRAIN_SAMPLES=3662109`, `SEQ_LEN=4096`.

Smoke versions add `-smoke` before `.sbatch`; they run 50 iterations, about
26.2M tokens total.

### 1.3B Scripts

These use `24L / 2048H / 5632F / 32h / 8kv`, 4 GPUs, `MBS=2`,
`GBS=128`, `RECOMPUTE=1`, checkpointing on, and default to about 60B tokens:
`TRAIN_SAMPLES=14648438`, `SEQ_LEN=4096`.

| Script | Optimizer | What it runs |
| --- | --- | --- |
| `_research/launch/transformer-pp-1.3b-gdn2-fullattn-3to1.sbatch` | AdamW | 1.3B 3:1 full-attention hybrid. |
| `_research/launch/transformer-pp-1.3b-gdn2-pure.sbatch` | AdamW | 1.3B pure GDN2. |
| `_research/launch/transformer-pp-1.3b-gdn2-fullattn-3to1-muon.sbatch` | NorMuon | 1.3B 3:1 full-attention hybrid with Muon. |
| `_research/launch/transformer-pp-1.3b-gdn2-pure-muon.sbatch` | NorMuon | 1.3B pure GDN2 with Muon. |

To use a bigger FineWeb-Edu dataset, only change the prefix:

```bash
export MEGATRON_DATA_PATH=/path/to/bigger_fineweb_edu_text_document
sbatch _research/launch/transformer-pp-1.3b-gdn2-fullattn-3to1.sbatch
```

To change the training token budget, override `TRAIN_SAMPLES`:

```bash
export TRAIN_SAMPLES=7324219   # about 30B tokens at seq len 4096
```

### Smoke Results

AdamW smoke results on 4 GPUs, 50 iters:

| Variant | Job | TFLOP/s/GPU | tokens/s/GPU | total tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 2:1 hybrid | `2360823` | `265.02` | `147,755.9` | `591,023.5` |
| pure GDN2 | `2360824` | `236.09` | `144,966.5` | `579,865.9` |
| 3:1 full-attn hybrid | `2360860` | `258.56` | `146,397.4` | `585,589.6` |

Muon smoke results on 4 GPUs, 50 iters:

| Variant | Job | TFLOP/s/GPU | tokens/s/GPU | total tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 2:1 hybrid | `2361454` | `255.20` | `142,276.4` | `569,105.8` |
| pure GDN2 | `2361455` | `229.34` | `140,820.0` | `563,280.0` |
| 3:1 full-attn hybrid | `2361499` | `251.05` | `142,146.8` | `568,587.3` |

Logs land under the submitting user:

```bash
/iopsstor/scratch/cscs/tr1eder/cler/_research/results/runs/
```
