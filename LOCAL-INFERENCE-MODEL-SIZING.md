# Local Inference Model Sizing (RTX 5060 Ti 16GB + 64GB RAM)

This report captures what the *largest* practical local LLM configurations look like on:

- GPU: NVIDIA RTX 5060 Ti 16GB (Blackwell, PCIe 5.0 x8, ~448 GB/s) ([TechPowerUp](https://www.techpowerup.com/gpu-specs/geforce-rtx-5060-ti-16-gb.c4292))
- System RAM: 64GB
- Target UX: ~16k context, >= 10 tok/s decode (interactive), with optional "push it" experiments for larger contexts/slower speeds

The intent is to use this as input to plan a new "Local Inference Lab" mode in this repo: a place to vary parameters, see throughput + memory tradeoffs, and build intuition about the hardware constraints (VRAM, KV cache, PCIe offload, batching, etc.).

## Executive Summary (What's "Max" On This Box?)

There are two different "max" answers depending on whether you mean **dense** or **MoE** models:

1. **Largest dense model that stays interactive at ~16k ctx**
   - Practical ceiling: **~24B @ Q4** (GGUF `Q4_K_M`-class) *fully resident in VRAM*.
   - Concrete datapoint on RTX 5060 Ti 16GB: **Devstral Small 2 24B `Q4_K_M`** at **24k ctx** gets **~9-11 tok/s decode**, uses **~15.2GB VRAM**, and avoids RAM offload. At **16k ctx** you should expect slightly better headroom/speed. ([Devstral Small 2 on 5060 Ti](https://www.reddit.com/r/LocalLLaMA/comments/1q7zywf/devstral_small_2_q4_k_m_on_5060_ti_16gb_and_zed/))

2. **Largest "overall" model that can still be fast (MoE "cheat code")**
   - Best current fit for "huge but still >=10 tok/s": **Qwen3-Coder-Next (80B-A3B MoE)** in GGUF with llama.cpp.
   - Concrete datapoint on RTX 5060 Ti 16GB: **Qwen3-Coder-Next `Q3_K_M`** at **32k ctx** reports **~16.9 tok/s decode** and **~154 tok/s prompt eval**, using **~15.1GB VRAM + ~30.2GB RAM**. With tuning (`-ngl 99`, `--n-cpu-moe`), the same author reports a sweet spot around **~296 tok/s prefill** and **~20 tok/s decode**. ([Reddit bench](https://www.reddit.com/r/LocalLLaMA/comments/1qwbmct/qwen3codernext_on_rtx_5060_ti_16_gb_some_numbers/), [Gist](https://gist.github.com/huytd/6b1e9f2271dd677346430c1b92893b57))

**Planning takeaway:** for a "largest model you can *use* locally" experience at your target, the likely "max" is **Qwen3-Coder-Next (80B-A3B)**. For "largest dense model that still feels OK", it's roughly **24B @ Q4** on 16GB VRAM.

## Why The Ceiling Exists (VRAM + KV Cache + PCIe)

### 1) Dense models want to fit entirely in VRAM

If a dense model's weights don't fit in VRAM, you typically end up paging weights over PCIe during decode. On the 5060 Ti, the bus is **PCIe 5.0 x8**, so while bandwidth is decent, it is still far below on-card GDDR7 bandwidth. ([TechPowerUp](https://www.techpowerup.com/gpu-specs/geforce-rtx-5060-ti-16-gb.c4292))

Result: **decode tok/s collapses** when you spill weights to system RAM for dense models.

### 2) Long context is mostly a KV-cache problem

As you raise `ctx`, the **KV cache** grows (often multiple GB), reducing the VRAM available for weights. That's why a 24B model can be viable at 8k but fail at 32k unless you quantize KV or reduce GPU residency.

Heuristic KV sizing (architecture-dependent, but directionally useful):

- KV memory scales roughly with: `O(n_layers * n_ctx * n_kv_heads * head_dim * bytes_per_elem * 2)` (K + V)
- `bytes_per_elem` is typically 2 for FP16, 1 for int8-ish, 0.5-ish for 4-bit-ish (depends on implementation)

For practical work, **treat KV as the main knob** for 16k-256k context experiments:

- Keep weights on GPU where possible
- Quantize KV (e.g. `--cache-type-k q8_0 --cache-type-v q4_0`) to recover VRAM at high `ctx`

### 3) MoE models change the game

MoE models can have a very large *total* parameter count while only activating a smaller subset per token. In practice this can make "80B total" feasible on a 16GB GPU *if* the runtime + model format can keep the hot path efficient (and sometimes offload experts sensibly to CPU/RAM).

### 4) Long-context scaling may be model-dependent (Qwen3-Next note)

Qwen3-Next models describe a hybrid architecture (including "Gated DeltaNet") intended to improve long-context efficiency relative to purely attention-based scaling. This likely contributes to why the Qwen3-Coder-Next benchmark reports similar decode tok/s at 32k vs 64k in llama.cpp. Treat this as a hypothesis to validate in your lab mode by sweeping context sizes and watching throughput/memory. ([Qwen3-Next report](https://qwenlm.github.io/blog/qwen3-next/), [HF tech report](https://huggingface.co/papers/2509.02851))

## Empirical Baselines On Similar Hardware

### A) Dense baseline: Devstral Small 2 (24B)

Observed on RTX 5060 Ti 16GB:

- Model: `Devstral-Small-2-24B-Instruct-2512` (GGUF), quant: `Q4_K_M`
- Context: 24k
- Decode: ~9-11 tok/s
- Prompt eval: ~648 tok/s
- VRAM used: ~15.2GB
- Notes: "Key is to fit everything in GPU VRAM so it won't offload anything to RAM" ([thread](https://www.reddit.com/r/LocalLLaMA/comments/1q7zywf/devstral_small_2_q4_k_m_on_5060_ti_16gb_and_zed/))

Implication for this repo's target (16k, >=10 tok/s): **this should be near the dense ceiling** if you want to avoid RAM offload entirely.

### B) MoE baseline: Qwen3-Coder-Next (80B-A3B)

Observed on RTX 5060 Ti 16GB (author's 32GB RAM box; you have 64GB):

- Model: `unsloth/Qwen3-Coder-Next-GGUF`, quant: `Q3_K_M`
- Context: 32k (and reported similar speed at 64k)
- Prompt eval: ~154-225 tok/s depending on task/context
- Decode: ~16.8-18.5 tok/s
- With tuning (`-ngl 99`, `--n-cpu-moe`): reported sweet spot around **~296 tok/s prefill** and **~20 tok/s decode** ([Reddit bench](https://www.reddit.com/r/LocalLLaMA/comments/1qwbmct/qwen3codernext_on_rtx_5060_ti_16_gb_some_numbers/), [Gist](https://gist.github.com/huytd/6b1e9f2271dd677346430c1b92893b57))

This is the strongest evidence that **"huge but still fast"** is achievable on your hardware.

## What "Largest" Means In Practice

For a learning lab, it helps to define "largest" along multiple axes:

- **Largest dense model without weight offload**: best proxy for "max quality per token without PCIe pain"
- **Largest total params that remain interactive**: MoE models win here
- **Largest context you can *use***: typically governed by KV cache type + VRAM overhead, plus prefill time

Recommendation: expose these as separate "leaderboards" or presets in the lab mode.

## Recommended Targets To Try First (For Your 16k / >=10 tok/s Goal)

### Tier 1 (Best "largest usable model"): Qwen3-Coder-Next (80B-A3B)

Why:

- Meets >=10 tok/s decode even at 32k+ in community benchmarks
- Uses system RAM in a way that still stays fast (unlike dense weight offload)

How to frame it in the lab:

- Baseline config: `Q3_K_M`, `ctx=16384` then scale to 32k/64k
- Show effects of `--n-cpu-moe` and `--fit on`
- Show prompt eval vs decode separately (agentic workflows care about prefill)

### Tier 2 (Largest dense option): Devstral Small 2 (24B)

Why:

- Dense 24B is near the VRAM ceiling for "no weight offload" on 16GB
- Reported ~9-11 tok/s decode at 24k ctx; likely >=10 at 16k

How to frame it in the lab:

- Baseline config: `Q4_K_M`, `ctx=16384`
- Show how increasing context pushes you into KV/cache tradeoffs

### Tier 3 (For "push it" / curiosity): bigger dense with heavy offload

Examples: 32B-70B dense quants with significant CPU/RAM involvement.

Expectation:

- Often falls below the "interactive" threshold due to PCIe + paging
- Still useful for learning: illustrates where throughput collapses and why

## llama.cpp Knobs Worth Surfacing (Because They Teach Hardware Limits)

### 1) `--fit on` / `--fit-ctx` / `--fit-n-gpu-layers`

Newer llama.cpp builds include an "autofit" mode to choose context and GPU layers to satisfy memory constraints. This is ideal for exploration because you can ask for 64k/128k and watch what gets sacrificed. ([llama.cpp discussion](https://github.com/ggml-org/llama.cpp/discussions/18049))

Related utilities:

- `llama-fit-params`: compute feasible settings given a GPU memory target ([llama.cpp discussion](https://github.com/ggml-org/llama.cpp/discussions/18049))

### 2) KV cache quantization: `--cache-type-k` and `--cache-type-v`

Example from the Qwen3-Coder-Next 5060 Ti bench:

```bash
--cache-type-k q8_0
--cache-type-v q4_0
```

This is a powerful lever for long context experiments. The lab mode should make it easy to toggle KV types and then show:

- VRAM consumption
- prompt eval tok/s
- decode tok/s
- quality regressions at long context (subjective but important)

### 3) Flash attention: `--flash-attn`

Flash attention can materially change both speed and how prefill scales with context. It's also a good "teaching knob" because it demonstrates when compute vs memory dominates.

### 4) Batch sizing: `--batch-size` and `--ubatch-size`

These primarily affect prompt evaluation throughput (prefill). A lab mode can demonstrate how to optimize "agentic" workloads that repeatedly ingest large prompts.

### 5) MoE offload knobs: `--n-cpu-moe`

Community benchmarks indicate `--n-cpu-moe` can materially affect Qwen3-Coder-Next throughput on limited VRAM setups. This is an important differentiator vs dense models.

## What A "Local Inference Lab" Mode Should Measure

To make this educational (and not just a wrapper around a model server), track:

- **Prompt eval tok/s (prefill)** and **decode tok/s** separately
- **VRAM used**, **RAM used** (at load + steady-state)
- **First-token latency** (time-to-first-token)
- **Context utilization** (prompt tokens, generated tokens)
- **CPU/GPU utilization** (even coarse: avg/max) to identify bottlenecks
- **Offload state**: GPU layers, KV cache types, MoE CPU experts, mmap/lock settings

If possible, store results in `outputs/` with a machine-readable schema so you can graph "tokens/sec vs ctx vs VRAM" over time.

## Suggested Experiment Matrix (Minimal But High Signal)

Start with a few "axes" and keep the cartesian product small:

1. **Model class**
   - Dense: Devstral Small 2 24B (`Q4_K_M`)
   - MoE: Qwen3-Coder-Next 80B-A3B (`Q3_K_M`)

2. **Context**
   - 4k, 8k, 16k, 32k (optional: 64k+)

3. **KV cache type**
   - Default (often FP16)
   - `K=q8_0, V=q4_0`
   - Optional: more aggressive V quant

4. **Autofit / offload**
   - `--fit off` with explicit `-ngl` / `--n-cpu-moe`
   - `--fit on` (observe what changes)

5. **Flash attention**
   - on/off

For each configuration, run a standardized prompt set:

- "Small context": 1-2k tokens prompt, generate 256
- "Medium context": 6-8k tokens prompt, generate 256
- "Large context": 14-16k tokens prompt, generate 256

## Planning Notes For Repo Integration

This repo already has:

- `api/` (FastAPI backend)
- `dashboard/` (React UI)

A clean way to integrate local inference experimentation:

- Use **`llama-server` as the engine** (OpenAI-ish HTTP surface + stable logs).
- Build an API-side "runner" that can:
  - start/stop a configured `llama-server` subprocess
  - stream logs + metrics over WebSocket
  - send chat/completion requests and stream tokens
  - record runs (config + measurements) to `outputs/`
- Build a dashboard page ("Local Inference") with:
  - model preset picker (Qwen3-Coder-Next, Devstral Small 2, etc.)
  - exposed knobs (ctx, KV cache types, flash attn, n-cpu-moe, batch sizes)
  - live charts (prefill tok/s, decode tok/s, VRAM/RAM)
  - a chat panel for qualitative evaluation

## References / Source Notes

- RTX 5060 Ti hardware specs (PCIe 5.0 x8, 16GB GDDR7, 448 GB/s): https://www.techpowerup.com/gpu-specs/geforce-rtx-5060-ti-16-gb.c4292
- Devstral Small 2 24B on RTX 5060 Ti (Q4_K_M, 24k ctx, ~9-11 tok/s): https://www.reddit.com/r/LocalLLaMA/comments/1q7zywf/devstral_small_2_q4_k_m_on_5060_ti_16gb_and_zed/
- Qwen3-Coder-Next on RTX 5060 Ti (Q3_K_M, 32k ctx, ~16.9 tok/s; tuning notes): https://www.reddit.com/r/LocalLLaMA/comments/1qwbmct/qwen3codernext_on_rtx_5060_ti_16_gb_some_numbers/
- Same Qwen3-Coder-Next benchmark captured as a gist (repro command + table): https://gist.github.com/huytd/6b1e9f2271dd677346430c1b92893b57
- llama.cpp `--fit` / autofit discussion and related tooling (`llama-fit-params`): https://github.com/ggml-org/llama.cpp/discussions/18049
- Qwen3-Next architecture report (Gated DeltaNet / long-context): https://qwenlm.github.io/blog/qwen3-next/
- Qwen3-Next technical report (HF Papers): https://huggingface.co/papers/2509.02851
