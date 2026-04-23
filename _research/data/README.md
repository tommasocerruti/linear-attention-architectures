# CLER Data Prep

The CLER baseline track uses a deterministic FineWeb-Edu prefix converted to
Megatron binary format with a LLaMA-2 SentencePiece tokenizer.

## LLaMA-2 Tokenizer Setup

The LLaMA-2 tokenizer is the shared SentencePiece tokenizer used by the
LLaMA-2 family; using `meta-llama/Llama-2-7b-hf` as the source is sufficient.
You only need `tokenizer.model`, not the model weights.

### Option A: download on Clariden

Use this if Hugging Face auth works on the cluster and your account has accepted
the LLaMA-2 access terms:

```bash
hf auth login
hf download meta-llama/Llama-2-7b-hf tokenizer.model \
  --local-dir /iopsstor/scratch/cscs/$USER/llama2-tokenizer

export LLAMA2_TOKENIZER_MODEL=/iopsstor/scratch/cscs/$USER/llama2-tokenizer/tokenizer.model
```

Older Hugging Face CLI installations may expose the same download path as
`huggingface-cli download` instead of `hf download`.

### Option B: download locally, then copy

Use this if cluster login cannot authenticate to Hugging Face:

```bash
hf download meta-llama/Llama-2-7b-hf tokenizer.model \
  --local-dir ./llama2-tokenizer

scp ./llama2-tokenizer/tokenizer.model \
  clariden:/iopsstor/scratch/cscs/$USER/llama2-tokenizer/tokenizer.model
```

Then, on Clariden:

```bash
export LLAMA2_TOKENIZER_MODEL=/iopsstor/scratch/cscs/$USER/llama2-tokenizer/tokenizer.model
```

## FineWeb-Edu 15B

Submit the conversion job from the repo root on Clariden:

```bash
export LLAMA2_TOKENIZER_MODEL=/iopsstor/scratch/cscs/$USER/llama2-tokenizer/tokenizer.model
sbatch _research/data/convert_fineweb_edu.sbatch
```

By default this streams
[`HuggingFaceFW/fineweb_edu_100BT-shuffled`](https://huggingface.co/datasets/HuggingFaceFW/fineweb_edu_100BT-shuffled),
writes the first document-boundary prefix that reaches 15B LLaMA-2 tokens, then
runs `tools/preprocess_data.py` with `Llama2Tokenizer`. The output is under:

```bash
_research/results/data/fineweb_edu/fineweb_edu_15b_llama2/
```

Use the printed Megatron prefix for training:

```bash
export MEGATRON_DATA_PATH=$PWD/_research/results/data/fineweb_edu/fineweb_edu_15b_llama2/fineweb_edu_15b_llama2_text_document
```

## FineWeb-Edu 100B

The same path scales to the full comparison subset:

```bash
FINEWEB_EDU_TOKENS=100000000000 FINEWEB_EDU_NAME=fineweb_edu_100b_llama2 \
  sbatch _research/data/convert_fineweb_edu.sbatch
```

The JSONL manifest next to the downloaded file records dataset id, split,
target tokens, written documents, and actual written token count.
