
Here’s a concrete “next set of upgrades” roadmap for **llm-testing** that’s tightly centered on (a) training bigger models on **16GB VRAM** and (b) **dashboard UX/UI** improvements, grounded in what’s already in your repo.

## What you already have (quick audit)

* **Two training stacks**:

  * **From-scratch GPT pretraining** (`experiments/pretraining`) with an educational implementation of attention + transformer blocks, plus a simple training loop. ([GitHub][1])
  * **QLoRA fine-tuning** via **Unsloth + TRL SFTTrainer** in the API (`api/routers/fine_tuning.py`) using **bf16**, **8-bit optimizer**, and **packing=True** (already strong choices for 16GB). ([GitHub][2])

* **A working dashboard** (`dashboard/`) that already does useful things:

  * Pretraining controls include dataset selection, checkpoint resume, and calls your `/estimate-vram` endpoint and shows warnings. ([GitHub][3])
  * Fine-tuning controls exist, but currently expose only a single model choice and a relatively small surface area of training knobs. ([GitHub][4])

* **VRAM estimator exists**, but is currently calibrated around fp32 and the “naive attention stores score matrices” assumption. That makes it conservative—and it becomes *very* pessimistic once you switch to Flash/SDPA attention. ([GitHub][5])

Also: you’re explicitly targeting an **RTX 5060 Ti 16GB (Blackwell)**; common board specs list **16GB GDDR7** and ~**448 GB/s** memory bandwidth. ([NVIDIA][6])

---

## The single biggest lever for “bigger models” on 16GB: fix attention memory

Right now your pretraining model’s `MultiHeadAttention` computes and stores the full `(B, H, T, T)` attention score tensor and weights (`attn_scores`, `attn_weights`). ([GitHub][1])

That is the classic “OOM at larger context lengths / model sizes” failure mode.

### Why it matters (concrete numbers)

If you run something like:

* batch `B=4`
* heads `H=12`
* seq length `T=1024`
* fp32 scores (4 bytes)

Then **just the attention scores** per layer cost:

`B * H * T * T * 4 bytes = 4 * 12 * 1024 * 1024 * 4 ≈ 192 MB per layer`

Across 12 layers that’s **~2.25 GB** *only for score matrices*, before you count the rest of activations, grads, optimizer states, etc. Your estimator explicitly accounts for this term. ([GitHub][5])

At `T=2048`, that term quadruples.

### Upgrade: add an SDPA/FlashAttention path (keep the educational path)

In PyTorch 2.x+, `torch.nn.functional.scaled_dot_product_attention` (SDPA) can dispatch to flash/memory‑efficient kernels when available, which avoids materializing the full attention matrix in the naive way.

Add a config knob:

* `attention_impl: Literal["manual","sdpa"] = "manual"`

Then in `MultiHeadAttention.forward`, do:

```python
import torch.nn.functional as F

# q,k,v: (B, H, T, D)
out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=self.dropout.p if self.training else 0.0,
    is_causal=True,
)
# out: (B, H, T, D) -> reshape to (B, T, d_out)
```

**Keep** your current “manual” implementation as the default for learning/visualization, but add “sdpa” as the “fast path” for scaling.

**Immediate payoff:** bigger `T`, bigger `B`, and/or bigger `emb_dim` become possible on 16GB, and tokens/sec should jump materially for the same settings.

---

## Pretraining loop upgrades that directly translate to “can fit bigger configs”

Your current pretraining training loop is clean but “baseline”: full precision, no grad accumulation, no AMP, no checkpointing. ([GitHub][7])

### 1) Add gradient accumulation (cheap, high impact)

Add CLI + API fields:

* `micro_batch_size` (what fits in VRAM)
* `grad_accum_steps` (to reach an effective batch)

Training loop sketch:

```python
optimizer.zero_grad(set_to_none=True)
for micro_step in range(grad_accum_steps):
    with autocast(...):
        loss = ...
        loss = loss / grad_accum_steps
    scaler.scale(loss).backward()

scaler.unscale_(optimizer)
clip_grad_norm_(...)
scaler.step(optimizer)
scaler.update()
```

This is the most reliable “make it fit” tool when VRAM is the constraint.

### 2) Mixed precision where it helps, but be deliberate

Two distinct wins:

* **Activation memory + throughput:** autocast to bf16/fp16 (even if weights remain fp32).
* **Parameter/optimizer memory:** requires more than autocast (e.g., 8-bit optimizer or true bf16 weights + fp32 master weights).

Given you’re already doing **bf16 + adamw_8bit** in fine-tuning, you can mirror that philosophy for “bigger pretraining configs” if desired. ([GitHub][2])

Pragmatic suggestion:

* Add `precision: "fp32"|"bf16"|"fp16"`:

  * default: `fp32` (learning mode)
  * “scale mode”: `bf16` + SDPA

* Enable TF32 matmuls on CUDA for speed (safe for training transformer baselines):

  * `torch.backends.cuda.matmul.allow_tf32 = True`
  * `torch.set_float32_matmul_precision("high")`

### 3) Gradient checkpointing toggle (for “I want a bigger model” days)

Add `gradient_checkpointing: bool`.

Implementation option that stays close to your code:

* Replace `nn.Sequential([...])` with a `ModuleList`
* In `GPTModel.forward`, iterate blocks and conditionally checkpoint each:

```python
from torch.utils.checkpoint import checkpoint

for block in self.trf_blocks:
    if self.cfg.gradient_checkpointing and self.training:
        x = checkpoint(block, x, use_reentrant=False)
    else:
        x = block(x)
```

This can be the difference between “medium fits” and “gpt2‑medium‑ish fits” on 16GB.

### 4) Turn on weight tying (free parameter + memory reduction)

Your model explicitly notes GPT-2 weight tying but keeps weights separate “for clarity.” ([GitHub][1])

Make it a flag:

* `tie_embeddings: bool = True` (default True in “scale mode”)

Implementation:

```python
self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
if cfg.tie_embeddings:
    self.out_head.weight = self.tok_emb.weight
```

This saves **one full vocab projection matrix** (for GPT‑2 small, ~38.6M params) and all the associated gradient/optimizer state.

---

## Make your VRAM estimator match reality after these upgrades

Today `estimate_vram_mb()` assumes:

* fp32 everywhere (`bytes_per_param = 4`)
* “attention score matrices exist in memory per layer” ([GitHub][5])

Once you add SDPA + checkpointing + bf16, that estimate becomes meaningfully wrong (and too pessimistic), which will mislead the dashboard warnings. ([GitHub][3])

Upgrade `estimate_vram_mb()` to accept:

* `precision`: fp32/bf16/fp16
* `optimizer`: adamw / adamw_8bit / paged_adamw_8bit
* `attention_impl`: manual/sdpa
* `gradient_checkpointing`: bool
* `tie_embeddings`: bool

Then:

* If `attention_impl == "sdpa"`, reduce or remove the `attention_scores_bytes` term (or scale it way down).
* If checkpointing is on, reduce activation factor (often 30–60% drop depending on how aggressively you checkpoint).
* If optimizer is 8-bit, reduce the optimizer state term significantly.

This makes the UX better *and* prevents users (you) from wasting time on “false OOM fear.”

---

## “Pretraining larger models” on 16GB: two practical meanings (and what to add)

You’ll hit a wall if “pretraining larger models” means “train multi‑B parameter models from scratch.”

But on a 16GB GPU, you *can* make meaningful progress in two directions:

### A) From-scratch scaling to “bigger than 124M”

With SDPA + grad accumulation + checkpointing + weight tying, you can add one or two larger presets beyond `medium`. For example:

* a “gpt2‑medium‑lite” around ~250–400M params, with:

  * smaller context to start (512–1024)
  * micro batch 1
  * checkpointing on
  * SDPA on

The point isn’t SOTA performance, it’s letting you explore:

* scaling laws qualitatively
* optimization stability at depth
* “loss curves at larger scale” with your existing dashboard

### B) Continued pretraining (domain adaptive) on 7B/14B via QLoRA

This is the **best** way to work with “larger models” on your hardware.

You already have the building blocks:

* Unsloth 4-bit load
* bf16
* 8-bit optimizer
* packing=True ([GitHub][2])

Add a new mode/track: **“Continued Pretraining”** (DAPT)

* Same plumbing as fine-tuning, but dataset is plain text (no instruction format needed)
* Goal: adapt base model to your domain corpus while keeping VRAM manageable

This will feel like “pretraining larger models” in practice, without pretending you’re going to train a foundation model from scratch on 16GB.

---

## Fine-tuning bigger base models on 16GB: changes I’d make in your API + dashboard

Your fine-tuning API already uses good defaults (bf16 + adamw_8bit + packing). ([GitHub][2])
The next jump is to generalize beyond one hardcoded model and add “fit‑to‑VRAM” controls.

### 1) Model registry + per-model LoRA target modules

Right now the UI only offers **Qwen2.5‑7B‑Instruct**. ([GitHub][4])

Add:

* `FINE_TUNING_MODELS = [{ name, hf_id, family, recommended_max_seq, recommended_lora_r, tags }]`

Backend:

* pick `target_modules` based on family (Qwen/Llama/Mistral/etc.)
* fall back to a heuristic “find linear layers with proj names” if family unknown

### 2) Add optimizer option: paged 8-bit for the “try 14B+” case

If you want to stretch into larger models on 16GB, system RAM is your friend.

Expose in config:

* `optim: "adamw_8bit" | "paged_adamw_8bit" | "adamw_torch"`

The “paged” optimizers are specifically designed to reduce VRAM pressure by paging optimizer states (tradeoff: slower). With **64GB RAM**, you’re well positioned to use that lever.

### 3) Expose gradient checkpointing & “Unsloth fast mode” as toggles

Your current code explicitly disables trainer patching via env var in at least one path. ([GitHub][2])

Make it user-facing:

* “Compatibility mode” (no patching)
* “Fast mode” (patching enabled)

Also add:

* `use_gradient_checkpointing: bool` (and possibly Unsloth’s specialized checkpointing option, if you want the long-context benefits it advertises) ([Unsloth][8])

### 4) Add a fine-tuning VRAM estimate endpoint

Mirror pretraining:

* `/api/fine-tuning/estimate-vram`
* show it in the fine-tuning controls the same way you do pretraining

This becomes *really* important once you add 14B+ models and longer contexts.

---

## Dashboard UX/UI improvements that will noticeably raise usability

You’ve already got the right core: live WebSockets + a clean control panel. ([GitHub][9])
Here are the upgrades that change the day-to-day experience:

### 1) Run persistence + “runs list”

Right now, metrics are effectively ephemeral (page refresh = you lose local history). You already store checkpoints with metadata on disk. ([GitHub][10])

Add:

* backend run IDs (`run_id`)
* store:

  * config
  * metrics time series (JSONL or sqlite)
  * key events (checkpoint saved, resumed, stopped, error)
* UI:

  * “Runs” sidebar
  * load prior run charts
  * compare two runs (overlay loss curves)

### 2) Add a **System Monitor** widget (GPU + CPU + RAM)

Expose an endpoint that returns:

* `gpu_name, vram_total, vram_used, vram_reserved`
* `gpu_utilization, temp`
* `system_ram_used/total`

Then show it persistently in the UI header. Your model manager already tracks some GPU memory stats for other tracks; build on that. ([GitHub][11])

### 3) “Auto-fit to VRAM” button

Best UX improvement for 16GB constraints:

Backend algorithm:

* given model config + context length + chosen precision/attention/checkpointing
* run a tiny forward/backward on dummy data with increasing micro-batch
* measure peak memory via `torch.cuda.max_memory_allocated()`
* return “max micro batch” + suggested grad accumulation for a target effective batch

UI:

* one click → fills in batch + grad accumulation safely

### 4) Better checkpoint browser

You already list checkpoints and warn on config mismatches. ([GitHub][3])

Improve:

* filter by corpus/config/context
* show metadata (loss, timestamp, tokens seen, precision, attention_impl)
* one-click “load for generation” / “resume training”

### 5) Make “advanced knobs” collapsible

For both pretraining and fine-tuning, split config UI:

* Basic: model, dataset, epochs, batch, LR
* Advanced: precision, attention_impl, grad accum, checkpointing, optimizer, warmup schedule

This keeps the UX approachable but powerful.

---

## Suggested PR sequence (minimizes churn, maximizes payoff)

1. **Pretraining SDPA attention** + keep manual attention behind a flag
   (Most direct speed/memory win.) ([GitHub][1])

2. **Pretraining grad accumulation + AMP toggle** (bf16/fp16)
   (Lets you scale batch/context safely.) ([GitHub][7])

3. **Gradient checkpointing + weight tying**
   (Unlocks bigger configs.) ([GitHub][1])

4. **Upgrade VRAM estimator** to reflect precision/SDPA/checkpointing/optimizer
   (Fixes UX and prevents “phantom OOM” warnings.) ([GitHub][5])

5. **Fine-tuning registry + optimizer options + VRAM estimate**
   (Enables “bigger models” fine-tuning properly.) ([GitHub][2])

6. **Run persistence + runs list + compare overlay**
   (Biggest UX jump for actual experimentation.)

---

If you want to prioritize *one* change that most directly enables “larger models on 16GB” in your current repo: implement **SDPA attention** in `MultiHeadAttention` as an alternate backend and wire it into the dashboard + VRAM estimator. Everything else compounds after that.

[1]: https://raw.githubusercontent.com/cmfunderburk/llm-testing/main/experiments/pretraining/model.py "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/cmfunderburk/llm-testing/main/api/routers/fine_tuning.py "https://raw.githubusercontent.com/cmfunderburk/llm-testing/main/api/routers/fine_tuning.py"
[3]: https://raw.githubusercontent.com/cmfunderburk/llm-testing/main/dashboard/src/components/pretraining/TrainingControls.tsx "https://raw.githubusercontent.com/cmfunderburk/llm-testing/main/dashboard/src/components/pretraining/TrainingControls.tsx"
[4]: https://raw.githubusercontent.com/cmfunderburk/llm-testing/main/dashboard/src/components/finetuning/FineTuningControls.tsx "https://raw.githubusercontent.com/cmfunderburk/llm-testing/main/dashboard/src/components/finetuning/FineTuningControls.tsx"
[5]: https://raw.githubusercontent.com/cmfunderburk/llm-testing/main/experiments/pretraining/config.py "https://raw.githubusercontent.com/cmfunderburk/llm-testing/main/experiments/pretraining/config.py"
[6]: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/ "https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/"
[7]: https://raw.githubusercontent.com/cmfunderburk/llm-testing/main/experiments/pretraining/train.py "https://raw.githubusercontent.com/cmfunderburk/llm-testing/main/experiments/pretraining/train.py"
[8]: https://unsloth.ai/blog/long-context "https://unsloth.ai/blog/long-context"
[9]: https://raw.githubusercontent.com/cmfunderburk/llm-testing/main/api/main.py "https://raw.githubusercontent.com/cmfunderburk/llm-testing/main/api/main.py"
[10]: https://raw.githubusercontent.com/cmfunderburk/llm-testing/main/experiments/pretraining/checkpoint.py "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/cmfunderburk/llm-testing/main/dashboard/src/types/index.ts "https://raw.githubusercontent.com/cmfunderburk/llm-testing/main/dashboard/src/types/index.ts"
