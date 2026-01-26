# LLM Learning Lab - Project Overview and Next Directions (2026-01-19)

This document is a "current state" snapshot of the repo: what exists, how the pieces fit together, what is actually working vs scaffolded, and concrete directions to take next given the lab's learning goals.

The framing assumption (based on your note):
- The pretraining module is close to a release candidate.
- The other tracks (fine-tuning / attention / probing / paper reproduction) are less developed and largely untested.

---

## 1) North Star (What This Repo Is For)

The repo is explicitly a learning vehicle, not a production pipeline.

Success criteria (as stated across `README.md` and `VISION.md`):
- Mental model: explain transformer architecture + training dynamics without reference material.
- Paper fluency: read a research paper and track methodology + claims.
- Experimental intuition: make predictions, run experiments, learn from surprises.

There are two "learning surfaces" in the codebase:

1) A from-scratch GPT pretraining lab (small models) with an interactive dashboard.
- Best for: training dynamics, debugging intuition, controlled experiments.

2) Large-model fine-tuning + mechanistic inspection tooling (Qwen2.5-7B via Unsloth).
- Best for: realistic LLM behavior changes, LoRA/QLoRA tradeoffs, interpretability at scale.

The main design challenge now is not "build more code"; it is to make these two surfaces cohere into a single learning lab experience.

---

## 2) Repository Map (Mental Model of the Code)

Top-level structure:

- `experiments/`
  - `pretraining/` (scratch GPT + datasets + training loop)
  - `fine_tuning/`, `learning_rate/`, `lora_rank/`, `forgetting/` (QLoRA experiments)
  - `attention/` (attention extraction + visualization)
  - `probing/` (activation extraction + analysis)
  - `paper_reproduction/` (currently Bayesian Geometry claim test)

- `api/` (unified FastAPI server)
  - `api/routers/pretraining.py` (pretraining training loop + WS stream)
  - `api/routers/attention.py` (attention extraction for Unsloth/HF models)
  - `api/routers/probing.py` (activation extraction for Unsloth/HF models)
  - `api/services/model_manager.py` (lazy loading of large models)

- `dashboard/` (React + TypeScript UI)
  - `PretrainingPage` uses WebSocket training stream
  - `AttentionPage` calls REST endpoints
  - `ProbingPage` calls REST endpoints

- `docs/` (concept docs, paper notes, retrospective)

- `outputs/` (experiment outputs)

A notable overlap:
- There is a legacy, standalone pretraining app under `experiments/pretraining/frontend/` (planned for removal) and `experiments/pretraining/api/` (legacy). The canonical stack is the root `api/` + `dashboard/`.

---

## 3) Pretraining Module (Release Candidate Candidate)

### What it is
A complete mini-stack that trains a decoder-only GPT from scratch and visualizes training live.

Core implementation (Python):
- `experiments/pretraining/model.py`
  - Decoder-only transformer blocks (pre-norm), causal attention, FFN, residuals.
  - Written in an educational style (lots of explanatory commentary).
- `experiments/pretraining/tokenizer.py`
  - GPT-2 BPE via `tiktoken`.
- `experiments/pretraining/data.py`
  - Sliding-window dataset, corpus registry, token caching to `.cache/*.pt`.
- `experiments/pretraining/train.py`
  - Training loop, LR warmup + cosine schedule, evaluation helpers.
- `experiments/pretraining/checkpoint.py`
  - Save/load checkpoints into `outputs/pretraining/` with metadata.
- `experiments/pretraining/download_corpora.py`
  - Downloads larger corpora (TinyStories, WikiText2, Shakespeare).

Interactive serving layer:
- `api/routers/pretraining.py`
  - TrainingManager that runs training asynchronously and streams metrics via WebSocket.
  - REST endpoints for status, start/pause/resume/stop, checkpoint listing, checkpoint deletion.
- `dashboard/src/pages/PretrainingPage.tsx`
  - Connects to `ws://localhost:8000/ws/training`.
  - Provides training controls + charts + metrics panels.

### Why this part is strong
- It is end-to-end. You can point at an entrypoint (`./run-dashboard.sh`) and see a cohesive system.
- The code is unusually educational: the commentary in `experiments/pretraining/model.py` and `train.py` is itself a learning artifact.
- The architecture is legible: scratch model + training loop + API wrapper + UI.
- It creates a short feedback loop. Small model configs (nano/small) allow iteration on a single GPU.

### Pretraining gaps to address before calling it "release candidate"
These are the main issues I see that are likely to matter quickly:

1) Validation loss is wired but not computed in the API training loop.
- `api/routers/pretraining.py` loads `val_loader` but never computes `val_loss`.
- UI and models include `val_loss`, but it will stay `null`.

2) The API includes a placeholder attention endpoint.
- `api/routers/pretraining.py` has `POST /api/pretraining/analyze/attention` described as a mock/uniform attention.
- This is fine as a stub, but it should be explicitly labeled or removed if "release" means correctness.

3) Checkpoint saving behavior is slightly inconsistent with "off by default".
- The API training loop always saves a final checkpoint (`checkpoint_final.pt`) even if checkpointing is off.
- This might be intended, but then the product promise should be "periodic checkpoints optional; final checkpoint always saved".

4) Duplicate/legacy pretraining frontend.
- Canonical frontend is `dashboard/`.
- `experiments/pretraining/frontend/` is legacy and planned for removal (to reduce confusion + maintenance).

5) Test coverage.
- If this is a candidate for "stable", it needs at least smoke tests: model forward shapes, tokenizer round-trip, dataloader shapes, checkpoint save/load, and a 50-step training run that decreases loss on `tiny`.

---

## 4) Unified API + Dashboard (Integration Layer)

### What is working today
- `api/main.py` mounts three tracks: pretraining, attention, probing.
- `dashboard/` offers three pages: Pretraining, Attention, Probing.
- Pretraining has real-time streaming via WebSocket (`/ws/training`).
- Attention and probing are synchronous REST calls.

### Why this is important
This integration layer is now the "product chassis" for the learning lab.

Even if most experiments remain CLI-first for now, having a stable API + UI provides:
- A consistent place to observe training dynamics.
- A consistent place to plug in future analysis tools.

### Integration mismatches / debt
- The dashboard has state management built around pretraining only (`TrainingContext`).
  - Attention/probing pages fetch directly with `fetch()` and local component state.
  - That is fine for MVP, but it means there is no unified "model session" concept across tracks.

---

## 5) Fine-Tuning Track (QLoRA) - Present but Not Integrated

### What exists
Track A code is in `experiments/fine_tuning/` plus supporting experiments:
- `experiments/fine_tuning/basic_qlora.py`
- `experiments/fine_tuning/loss_curve_analysis.py`
- `experiments/learning_rate/experiment.py`
- `experiments/lora_rank/experiment.py`
- `experiments/forgetting/experiment.py`

These are designed as hypothesis-driven experiments and are well-structured for learning.

### Current limitations
- They are not integrated with the dashboard.
- They are (per docs and placeholders) not run yet, and results sections are largely empty.
- The learning-rate README appears to reference an incorrect module path (it mentions `experiments.fine_tuning.learning_rate.experiment`, but the actual path is `experiments.learning_rate.experiment`).

### Practical direction
If you want the pretraining module to be "release-like" and everything else to be "experimental", that split is reasonable.

But if you want a unified learning lab experience, the fine-tuning track likely needs either:
- A thin dashboard integration (start a fine-tune run, stream metrics, browse outputs), or
- A deliberate decision to keep fine-tuning as CLI-only and treat the dashboard as a separate product (pretraining lab).

---

## 6) Attention Track (Extraction + Visualization) - Partially Wired

### What exists
- Extraction: `experiments/attention/extract.py`
- Visualization: `experiments/attention/visualize.py`
- API: `api/routers/attention.py` (loads Unsloth model via `api/services/model_manager.py`)
- UI: `dashboard/src/pages/AttentionPage.tsx`

The approach is hook-based:
- Enables `model.config.output_attentions = True`.
- Hooks each attention module and caches returned attention weights.

### What is good
- Clear, explicit tensor-shape documentation for Qwen (GQA considerations).
- The API path is clean: UI -> REST -> model_manager -> experiments extractor.

### Gaps / risk
- Not yet validated end-to-end on GPU.
- The compare experiment (`experiments/attention/compare_experiment.py`) is incomplete:
  - It explicitly skips extracting from the base model due to memory constraints.
  - This blocks the core promised comparison (base vs fine-tuned).

### Direction
A good next step is to make comparison feasible by design:
- Run sequentially and persist intermediate results to disk (attention tensors per layer/head), then compare offline.
- Or compare on a smaller model (e.g. Qwen2.5-1.5B) first to prove the pipeline.

---

## 7) Probing Track (Activations / Residual Stream) - Has a Core Bug

### What exists
- Extraction: `experiments/probing/extract.py`
- Analysis experiment: `experiments/probing/run_analysis.py`
- API: `api/routers/probing.py`
- UI: `dashboard/src/pages/ProbingPage.tsx`

### The critical issue
The code and docs intend to support three extraction points:
- `pre_attn`
- `post_attn`
- `post_ffn`

But the current extractor only captures:
- `pre_attn`
- `post_ffn`

Consequences:
- `ActivationOutput.get_layer_diff()` expects `post_attn`, so it cannot work today.
- The paper reproduction experiment (`experiments/paper_reproduction/bayesian_geometry/experiment.py`) requests `post_attn` and will likely fail.
- The API's `/api/probing/layer-diff` endpoint currently returns an "approximation" and duplicates the same stats for attention and FFN.

### Direction (highest-leverage fix outside pretraining)
Implement a real `post_attn` capture path.

Conceptually, for decoder-only transformer blocks (Llama/Qwen style):
- Capture `pre_attn` as layer input residual.
- Capture the raw attention output from `layer.self_attn`.
- Compute `post_attn = pre_attn + attn_output` (plus dropout if applicable).
- Capture `post_ffn` as layer output.

Once this is fixed:
- Representation probing becomes real.
- The Bayesian geometry claim test becomes runnable.
- The dashboard probing page can evolve from "summary stats" to real component-wise contribution analysis.

---

## 8) Paper Reproduction Track (Bayesian Geometry) - Blocked by Probing

### What exists
- Paper notes: `docs/papers/bayesian-geometry-attention.md`
- Experiment scaffold: `experiments/paper_reproduction/bayesian_geometry/experiment.py`

### Current status
- Strong conceptual groundwork.
- Execution likely blocked by the missing `post_attn` extraction.

### Direction
Once probing is fixed, this track becomes a great "integration test" for the learning lab:
- It forces coherence between attention extraction, activation extraction, and analysis.
- It yields a concrete artifact: "does this claim hold on a real instruction-tuned model?"

---

## 9) What to Do Next (A Practical Roadmap)

This is structured as options rather than a single mandate. The best choice depends on which learning loop you want to emphasize.

### Option 1 (Recommended): Stabilize Pretraining as the Core Product, Then Pull Other Tracks Into It

Premise:
- Small-model pretraining is the fastest iteration loop.
- It is the best place to build experimental intuition.

Near-term (1-3 days):
- Add val loss computation in the API training loop (or remove it everywhere).
- Make checkpoint behavior explicit (periodic optional; final always saved, or truly opt-in).
- Canonical UI is `dashboard/`; remove `experiments/pretraining/frontend/`.
- Add smoke tests for scratch GPT (tokenizer, dataloader, model forward, checkpoint round-trip).

Short-term (1-2 weeks):
- Add "analysis of scratch GPT" into the dashboard:
  - Attention heatmap for the scratch model.
  - Activation norms across layers.
  - Per-layer contribution charts.

Why this is aligned with the lab goals:
- The fastest way to build mechanistic intuition is to train small models and instrument them deeply.

### Option 2: Make the Unsloth/Qwen Tracks Real First (Run + Fix + Document Results)

Premise:
- The lab also wants intuition about modern LLMs, not only toy GPTs.

Near-term (1-2 weeks):
- Fix probing extractor (`post_attn`).
- Run these experiments once on GPU:
  - `python -m experiments.fine_tuning.basic_qlora`
  - `python -m experiments.fine_tuning.loss_curve_analysis`
  - `python -m experiments.attention.compare_experiment` (or a smaller version)
  - `python -m experiments.probing.run_analysis`
  - `python -m experiments.paper_reproduction.bayesian_geometry.experiment`
- Write results into the existing RESULTS/LEARNINGS sections.

Short-term follow-up:
- Make sure the API endpoints (attention/probing) match the experiment tooling outputs.

Why this is aligned:
- It grounds the scaffolding in empirical outcomes and closes the "hypothesis -> evidence" loop.

### Option 3: Treat This Repo as Two Products (Pretraining Lab + LLM Learning Lab)

Premise:
- The pretraining dashboard is becoming a real tool.
- The fine-tuning + interpretability tracks are research notebooks.

Actions:
- Make `./run-dashboard.sh` + `api/` + `dashboard/` the official "pretraining lab".
- Keep the Unsloth experiments as a separate CLI workflow with docs.
- De-emphasize the unified dashboard as the home for everything.

Why this might be correct:
- It avoids building a lot of complex UI glue.
- It keeps learning loops separate and reduces scope creep.

---

## 10) Testing Strategy (Minimum Viable Confidence)

If pretraining is "release candidate", a lightweight test plan provides a lot of leverage.

### Unit-ish tests (fast, CPU)
- Tokenizer round-trip:
  - `decode(encode(text)) == text` for representative strings.
- Data pipeline shapes:
  - `get_dataloader(...).batch['input_ids'].shape == (batch, context)`.
- Model forward shapes:
  - `(batch, seq_len, vocab_size)`.
- Checkpoint save/load:
  - After saving and loading, `state_dict` keys match; a forward pass works.

### Smoke tests (GPU optional)
- A 50-200 step run on `tiny` or `verdict` should reduce loss.
- Dashboard connects to WebSocket and renders metrics.

### Integration checks (manual is OK at first)
- `./run-dashboard.sh` starts both servers.
- Start training in UI; verify:
  - loss chart updates
  - tokens/sec moves
  - final checkpoint appears

---

## 11) Discussion Prompts (To Decide the Next Direction)

To pick the best next step, I would ask:

1) Which learning loop do you want to optimize for right now?
- "fast iteration on fundamentals" -> pretraining-first.
- "real-model behavior change" -> QLoRA-first.

2) Do you want the dashboard to become the unified home for all tracks?
- If yes, we should define a shared concept of "active model session" (scratch GPT vs Qwen) and unify state.
- If no, keep the dashboard pretraining-focused and keep other tracks CLI-driven.

3) How much engineering effort should go into polish vs new experiments?
- If the goal is learning velocity, prioritize correctness + minimal ergonomics.
- If the goal is a sharable tool, prioritize tests + docs + packaging + cleanup.

---

## Appendix: Notable Files (Quick Links)

Pretraining core:
- `experiments/pretraining/model.py`
- `experiments/pretraining/train.py`
- `experiments/pretraining/data.py`
- `experiments/pretraining/tokenizer.py`
- `api/routers/pretraining.py`
- `dashboard/src/pages/PretrainingPage.tsx`

Attention/probing API:
- `api/services/model_manager.py`
- `api/routers/attention.py`
- `api/routers/probing.py`

Unsloth experiments:
- `experiments/fine_tuning/basic_qlora.py`
- `experiments/attention/extract.py`
- `experiments/probing/extract.py`
- `experiments/paper_reproduction/bayesian_geometry/experiment.py`
