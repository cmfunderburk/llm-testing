# Ralph Loop Startup: LLM Pretraining Lab

Copy and paste the command below to start the ralph-loop.

---

## Pre-flight Checklist

Before starting:
- [ ] Virtual environment activated: `source .venv/bin/activate`
- [ ] Dependencies installed: `pip install torch fastapi uvicorn websockets tiktoken pyyaml`
- [ ] Node.js available: `node --version` (v18+ recommended)
- [ ] On correct branch: `git branch`
- [ ] Working directory clean: `git status`
- [ ] PRD reviewed: `experiments/pretraining/PRD-PRETRAINING-LAB.json`

---

## Startup Prompt

```
/ralph-loop:ralph-loop "Execute experiments/pretraining/PRD-PRETRAINING-LAB.json systematically.

This is a LEARNING PROJECT building an LLM pretraining platform from scratch.

PHASE INSTRUCTIONS:
- Phase 1 (Core Architecture): Implement GPT model following Raschka Ch 4. Reference docs/book-chapters/text/04-implementing-gpt-model.txt for architecture details. Include educational comments explaining each component.
- Phase 2 (Training Infrastructure): Implement training loop following Ch 5. Reference docs/book-chapters/text/05-pretraining-on-unlabeled-data.txt.
- Phase 3 (Backend API): FastAPI + WebSocket. Keep API simple and well-documented.
- Phase 4 (Frontend Foundation): React + TypeScript with Vite. Create in experiments/pretraining/frontend/
- Phase 5 (Real-time Dashboard): D3.js or Recharts for visualizations. Focus on usability.
- Phase 6 (Analysis Tools): Interactive attention/activation visualizations.

CONSTRAINTS:
- ASK before proceeding to each new phase
- Educational comments required in model code
- Checkpoint storage OFF by default
- Model configs: nano (~10M), small (~50M), medium (~124M)
- Test with 'verdict' corpus for quick iterations

ESCALATION TRIGGERS - STOP if:
- Frontend build fails
- NaN losses during training
- WebSocket connection fails after 3 retries
- >20 iterations without progress on a phase

Track progress via git commits + experiments/pretraining/progress.txt" --max-iterations 50 --completion-promise "PRETRAINING-LAB-COMPLETE"
```

---

## Resume Prompt (for interrupted loops)

```
/ralph-loop:ralph-loop "Resume experiments/pretraining/PRD-PRETRAINING-LAB.json execution.

Check current state:
1. Read experiments/pretraining/progress.txt for last known state
2. Check PRD for feature 'passes' values and phase status
3. Review recent git commits: git log --oneline -10

Continue from where execution stopped. ASK before proceeding to a new phase." --max-iterations 50 --completion-promise "PRETRAINING-LAB-COMPLETE"
```

---

## Phase-Specific Continuation Prompts

### After Phase 1 (Core Architecture) Review:
```
/ralph-loop:ralph-loop "Continue PRD-PRETRAINING-LAB.json to Phase 2: Training Infrastructure.

Phase 1 verified complete:
- GPT model forward pass works
- Tokenizer encodes/decodes correctly
- Data pipeline produces batches

Proceed with training loop implementation." --max-iterations 50 --completion-promise "PRETRAINING-LAB-COMPLETE"
```

### After Phase 2 (Training Infrastructure) Review:
```
/ralph-loop:ralph-loop "Continue PRD-PRETRAINING-LAB.json to Phase 3: Backend API.

Phase 2 verified complete:
- Training runs 1 epoch without errors
- Checkpoints save/load correctly
- CLI accepts configuration

Proceed with FastAPI server implementation." --max-iterations 50 --completion-promise "PRETRAINING-LAB-COMPLETE"
```

### After Phase 3 (Backend API) Review:
```
/ralph-loop:ralph-loop "Continue PRD-PRETRAINING-LAB.json to Phase 4: Frontend Foundation.

Phase 3 verified complete:
- API endpoints respond
- WebSocket accepts connections
- Training control endpoints work

Proceed with React + TypeScript frontend." --max-iterations 50 --completion-promise "PRETRAINING-LAB-COMPLETE"
```

### After Phase 4 (Frontend Foundation) Review:
```
/ralph-loop:ralph-loop "Continue PRD-PRETRAINING-LAB.json to Phases 5 & 6 (parallel eligible).

Phase 4 verified complete:
- Frontend builds and serves
- WebSocket connects to backend
- Basic UI renders

Proceed with real-time dashboard (Phase 5) and analysis tools (Phase 6). These can be developed in parallel." --max-iterations 50 --completion-promise "PRETRAINING-LAB-COMPLETE"
```

---

## Key Files

| File | Purpose |
|------|---------|
| `experiments/pretraining/PRD-PRETRAINING-LAB.json` | Full PRD specification |
| `experiments/pretraining/progress.txt` | Append-only progress log |
| `experiments/pretraining/RETROSPECTIVE.md` | Learning retrospective template |
| `docs/book-chapters/text/04-implementing-gpt-model.txt` | GPT architecture reference |
| `docs/book-chapters/text/05-pretraining-on-unlabeled-data.txt` | Pretraining reference |
| `VISION.md` | Overall project vision |

---

## Expected Outputs

### Phase 1: Core Architecture
- `experiments/pretraining/model.py` - GPT model implementation
- `experiments/pretraining/tokenizer.py` - BPE tokenizer wrapper
- `experiments/pretraining/data.py` - Dataset and DataLoader
- `experiments/pretraining/config.py` - Model configurations

### Phase 2: Training Infrastructure
- `experiments/pretraining/train.py` - Training loop and CLI
- `experiments/pretraining/checkpoint.py` - Save/load utilities
- `experiments/pretraining/generate.py` - Text generation
- `outputs/pretraining/` - Training outputs directory

### Phase 3: Backend API
- `experiments/pretraining/api/` - FastAPI application
- `experiments/pretraining/api/main.py` - Server entry point
- `experiments/pretraining/api/routes/` - API endpoints
- `experiments/pretraining/api/websocket.py` - WebSocket handler

### Phase 4: Frontend Foundation
- `experiments/pretraining/frontend/` - React application
- `experiments/pretraining/frontend/src/` - Source code
- `experiments/pretraining/frontend/package.json` - Dependencies

### Phase 5: Real-time Dashboard
- Components in `frontend/src/components/dashboard/`
- Loss curve chart
- Metrics panel
- Training controls

### Phase 6: Analysis Tools
- Components in `frontend/src/components/analysis/`
- Checkpoint browser
- Attention heatmap
- Activation visualization

---

## Progress Tracking Template

Create `experiments/pretraining/progress.txt` at loop start:

```markdown
## Progress Log: LLM Pretraining Lab

---

### Phase 1: Core Architecture

#### Completed
- [timestamp] FEAT-001: Multi-Head Attention - [description]

#### Issues Encountered
- [timestamp] ISSUE: [description]
  - Root cause: [cause]
  - Resolution: [fix]

#### Skipped/Deferred
- [timestamp] SKIPPED: [feature] - [reason]

---

### Phase 2: Training Infrastructure
[same structure]

---

## Summary Statistics
| Metric | Count |
|--------|-------|
| Features completed | X/28 |
| Phases completed | X/6 |
```

---

## Verification Commands Quick Reference

```bash
# Phase 1 gate
python -c "from experiments.pretraining.model import GPTModel; print('Model OK')"
python -c "from experiments.pretraining.tokenizer import Tokenizer; t = Tokenizer(); print(t.decode(t.encode('test')))"
python -c "from experiments.pretraining.data import get_dataloader; print('Data OK')"

# Phase 2 gate
python -m experiments.pretraining.train --config nano --epochs 1 --corpus verdict

# Phase 3 gate
curl http://localhost:8000/api/health

# Phase 4 gate
curl http://localhost:3000 | grep -q 'Pretraining Lab' && echo "Frontend OK"

# Phases 5-6: Manual verification in browser
```

---

*Generated 2026-01-12 for LLM Learning Lab pretraining track*
