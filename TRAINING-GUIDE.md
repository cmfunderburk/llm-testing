# GPT Pretraining Guide

This guide focuses on the expanded pretraining surface in the LLM Learning Lab.

Goal: build intuition for transformer training dynamics, not production-scale training.

## Quick Start

### Dashboard path (recommended)

```bash
./run-dashboard.sh
```

Open the frontend URL and go to **Pretraining**.

### CLI path

```bash
source .venv/bin/activate
python -m experiments.pretraining.train --config nano --corpus verdict --epochs 3
```

## Dashboard vs CLI

| Surface | Best For | Notes |
|---------|----------|-------|
| Dashboard (`/api/pretraining/*`) | Interactive runs, VRAM planning, checkpoint resume, run comparisons | Most feature-complete pretraining workflow |
| CLI (`python -m experiments.pretraining.train`) | Fast scripted smoke tests and simple experiments | Narrower config surface than dashboard |

## Model Configurations

| Config | Params (approx) | Default Context | Typical Use |
|--------|------------------|-----------------|-------------|
| `nano` | ~10M | 256 | Pipeline validation, quick iteration |
| `small` | ~50M | 512 | Medium-speed experiments |
| `medium` | ~124M | 1024 | Serious learning runs |
| `large` | ~204M | 1024 | Capacity experiments (VRAM-sensitive) |
| `xlarge` | ~355M | 1024 | Stretch experiments, requires careful settings |

Recommendation: start with `nano`, then scale one step at a time.

## Corpus Options

### Built-in quick corpora

- `verdict`: tiny literary corpus for fast sanity checks
- `tiny`: minimal smoke-test corpus

### Downloadable corpora

```bash
python -m experiments.pretraining.download_corpora --list
python -m experiments.pretraining.download_corpora tinystories
python -m experiments.pretraining.download_corpora wikitext2
python -m experiments.pretraining.download_corpora shakespeare
python -m experiments.pretraining.download_corpora pg19_small
python -m experiments.pretraining.download_corpora pg19_docs_small
```

Notable options:

- `tinystories`: recommended first meaningful corpus
- `wikitext2`: compact benchmark-style corpus
- `shakespeare`: stylized corpus for qualitative sampling checks
- `wikipedia_ga_intros`: prepared locally via project script
- PG-19 families:
  - flat text splits (`pg19_*`)
  - document-boundary JSONL splits (`pg19_*_docs`)
  - normalized variants (`*_normalized`)

## Suggested Run Progression

### Run 1: Smoke test

```bash
python -m experiments.pretraining.train --config nano --corpus verdict --epochs 3
```

Expected behavior:

- rapid train loss drop
- obvious memorization on a tiny corpus

### Run 2: First meaningful run

```bash
python -m experiments.pretraining.download_corpora tinystories
python -m experiments.pretraining.train --config nano --corpus tinystories --epochs 1
```

Expected behavior:

- slower, smoother loss descent
- generation quality improves without direct memorization

### Run 3: Controlled scaling

Train on the same corpus while changing config size (`nano` -> `small` -> `medium`) and compare:

- final train/val losses
- generation quality
- throughput and wall-clock time

Use dashboard run-history overlays to compare curves directly.

## Advanced Dashboard Controls (Pretraining)

The dashboard exposes controls not available in the basic CLI path:

- Optimizer: `adamw`, `adamw_8bit`, `paged_adamw_8bit`
- Precision: `fp32`, `bf16`, `fp16`
- Attention backend: `manual` vs `sdpa`
- Gradient checkpointing toggle
- Tied embeddings toggle
- Resume from checkpoint + config mismatch warnings
- Manual "save now" checkpoint
- Intermediate validation frequency controls

Use `estimate-vram` feedback before launching high-memory runs.

## Metrics That Matter

### Training loss

- flat/high: likely LR or data/config issue
- noisy spikes: often too aggressive LR or unstable mixed precision
- very low on tiny corpus: memorization, not necessarily generalization

### Validation loss

- tracks train loss: decent fit/generalization signal
- diverges while train loss drops: overfitting

### Tokens/second

- proxy for training throughput
- useful for comparing config and optimization settings

### Sample generations

- fastest qualitative check of whether improvements are meaningful
- review samples at similar steps across runs for fair comparison

## Common Issues

### Out of memory

- reduce `batch_size` and/or `context_length`
- drop model size (`xlarge` -> `large` -> `medium` ...)
- switch `attention_impl` to `sdpa`
- enable `gradient_checkpointing`
- use `bf16` when supported

### Resume instability

When resuming, keep `corpus`, `batch_size`, `context_length`, and optimizer aligned unless you intentionally want a changed trajectory.

### Slow tokenization on large corpora

The pipeline supports chunked tokenization + caching. First run can be long; subsequent runs should reuse token cache unless source files changed.

## Learning Exercises

1. Memorization vs generalization: compare `verdict` and `tinystories` runs at similar step counts.
2. Attention backend tradeoff: run same config with `manual` and `sdpa`; compare memory and throughput.
3. Precision tradeoff: compare `fp32` vs `bf16` stability and speed on one corpus/config.
4. Checkpoint resume behavior: pause, save checkpoint, resume, and verify curve continuity.

## Next Steps

1. Move to **Attention** with a trained checkpoint to inspect head behavior.
2. Move to **Probing** to inspect layerwise representation changes.
3. Use **Fine-Tuning** to contrast adaptation dynamics against pretraining dynamics.
