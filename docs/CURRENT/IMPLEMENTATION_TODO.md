# LLM-Testing Implementation Tracker

Last updated: 2026-02-13
Primary source docs:
- `docs/CURRENT/directions.md`
- `docs/CURRENT/PROJECT_OVERVIEW_AND_DIRECTIONS.md`

## Goal
Ship the roadmap from `docs/CURRENT/directions.md` in a staged way that keeps the app usable at each step and continuously improves 16GB VRAM viability.

## How To Keep This Updated
- Update checkboxes at the end of each merged change.
- Add one short note in `Progress Log` for what landed and what is next.
- If scope changes, update the relevant `Acceptance Criteria` before implementation.

## Overall Milestones
- [x] M1: Pretraining scale controls are fully plumbed (API + UI + estimator contract)
- [x] M2: SDPA attention path implemented and selectable
- [x] M3: Pretraining loop supports grad accumulation + AMP + checkpointing toggles
- [x] M4: VRAM estimator reflects precision/attention/checkpointing/optimizer/tied embeddings
- [x] M5: Fine-tuning model registry + optimizer/fast-mode controls + VRAM estimate
- [x] M6: Run persistence + runs list + run comparison UX

## Phase 1: Plumbing And UI Scaffolding

### 1.1 Pretraining config contract
- [x] Add fields to backend `TrainingConfig`:
  - `attention_impl` (`manual|sdpa`)
  - `precision` (`fp32|bf16|fp16`)
  - `grad_accum_steps` (int >= 1)
  - `gradient_checkpointing` (bool)
  - `tie_embeddings` (bool)
- [x] Add matching fields to frontend `TrainingConfig` in `dashboard/src/types/index.ts`
- [x] Ensure defaults are non-breaking and preserve current behavior

Acceptance criteria:
- `POST /api/pretraining/start` accepts new fields without breaking existing clients.
- Existing configs that do not send new fields still run.

### 1.2 VRAM estimate API contract expansion
- [x] Extend `GET /api/pretraining/estimate-vram` inputs to accept:
  - `precision`, `attention_impl`, `gradient_checkpointing`, `tie_embeddings`
- [x] Keep old query shape supported
- [x] Return remains backward-compatible with current UI

Acceptance criteria:
- Current UI call still works unchanged.
- New UI call with advanced fields also works.

### 1.3 Pretraining controls UX split
- [x] Add Basic section (current core controls)
- [x] Add collapsible Advanced section:
  - precision
  - attention implementation
  - gradient accumulation steps
  - gradient checkpointing
  - tie embeddings
- [x] Include effective batch display (`batch_size * grad_accum_steps`)

Acceptance criteria:
- Users can configure advanced options without cluttering default flow.
- Saved/loaded config remains stable during active run sync.

## Phase 2: SDPA + Core Memory Wins

### 2.1 Attention backend toggle in model
- [x] Add `attention_impl` to GPT config
- [x] Implement SDPA path in `MultiHeadAttention.forward`
- [x] Keep manual path for educational mode

Acceptance criteria:
- Both paths run for same shapes.
- SDPA path is causal and dropout-safe.

### 2.2 Embed tying
- [x] Add `tie_embeddings` model flag
- [x] Tie output head to token embeddings when enabled

Acceptance criteria:
- Parameter count reflects tying.
- Forward pass and checkpoint load/save still work.

## Phase 3: Training Loop Scaling Controls

### 3.1 Gradient accumulation
- [x] Add micro-step loop in API pretraining trainer
- [x] Scale loss by accumulation steps
- [x] Clip and step once per accumulated update

### 3.2 Mixed precision
- [x] Add autocast path for bf16/fp16
- [x] Add GradScaler logic for fp16 path
- [x] Keep fp32 path unchanged

### 3.3 Gradient checkpointing
- [x] Convert transformer block container to support per-block checkpointing
- [x] Enable only for training mode and when configured

Acceptance criteria:
- No regression with default settings.
- Advanced settings run at least one short smoke run.

## Phase 4: VRAM Estimator Reality Update
- [x] Update estimator internals for new knobs
- [x] Scale/reduce attention matrix term for SDPA
- [x] Adjust activation estimate for checkpointing
- [x] Reflect precision and embedding tying in model/grad/optimizer terms

Acceptance criteria:
- Estimator trend matches observed behavior directionally.
- Warning thresholds stay sensible for 16GB.

## Phase 5: Fine-Tuning Scale UX
- [x] Introduce model registry (not single hardcoded option)
- [x] Family-aware LoRA target module selection
- [x] Optimizer choice (`adamw_8bit`, `paged_adamw_8bit`, `adamw_torch`)
- [x] Fast mode vs compatibility mode toggle
- [x] Add fine-tuning VRAM estimate endpoint + UI widget

## Phase 6: Run Persistence And Comparison
- [x] Persist run metadata + metrics history
- [x] Add runs list UI
- [x] Load historical runs and overlay comparisons

## Risks / Watchouts
- API and dashboard config drift (must keep types in lockstep).
- Added knobs without actual backend behavior (mark clearly until implemented).
- VRAM estimator changes may affect user trust if not calibrated against observed runs.
- SDPA availability differences across PyTorch/CUDA environments.

## Progress Log
- 2026-02-13: Tracker created. Next: implement Phase 1.1 and 1.3 plumbing with non-breaking defaults.
- 2026-02-13: Completed Phase 1 scaffolding pass (new pretraining config fields, expanded VRAM estimate query contract, advanced controls UI with effective batch display).
- 2026-02-13: Completed Phase 2 model updates (manual/SDPA attention toggle + tied embeddings), wired API model construction to advanced config fields, and smoke-tested forward pass for both attention paths.
- 2026-02-13: Implemented Phase 3.1/3.2 in API pretraining loop (grad accumulation + AMP with fp16 scaler, bf16/fp16 autocast, TF32 enablement on CUDA). Remaining Phase 3 work: gradient checkpointing.
- 2026-02-13: Completed Phase 3.3 by enabling per-block gradient checkpointing in `GPTModel.forward`; verified train/eval passes with checkpointing enabled.
- 2026-02-13: Completed Phase 4 estimator updates (optimizer-aware endpoint/UI controls, SDPA/checkpointing scaling, precision/tie-aware memory terms, and stronger warning guidance).
- 2026-02-13: Completed Phase 5 fine-tuning scale UX (model registry endpoint/UI, family-aware LoRA targets with fallback inference, optimizer + fast mode + checkpointing controls, and new fine-tuning VRAM estimate endpoint/widget).
- 2026-02-13: Completed Phase 6 for pretraining runs (persistent `run_id` metadata/history on disk, runs list/detail API, dashboard run history panel, and historical loss-curve overlays for run comparison).
