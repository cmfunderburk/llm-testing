# Repository Guidelines

## Project Structure & Module Organization
- `api/`: FastAPI backend (`api/main.py`) with routers in `api/routers/` and shared services in `api/services/`.
- `dashboard/`: React + TypeScript frontend (Vite). Main code lives in `dashboard/src/` (`components/`, `pages/`, `context/`, `hooks/`).
- `experiments/`: Python experiment tracks (fine-tuning, pretraining, attention, probing, paper reproduction) run as modules.
- `docs/`: concept notes, retrospectives, and reference material.
- `outputs/`: generated artifacts (checkpoints, logs, plots). Treat as runtime data, not source.
- `qwen-finetune/`: separate fine-tuning workspace and scripts.

## Build, Test, and Development Commands
- `uv sync`: install Python dependencies from `pyproject.toml`/`uv.lock`.
- `source .venv/bin/activate`: activate local Python environment.
- `./run-dashboard.sh`: start API (`:8000`) and frontend (`:5173`) together.
- `uvicorn api.main:app --reload --port 8000`: run backend only.
- `cd dashboard && npm install && npm run dev`: run frontend only.
- `cd dashboard && npm run lint && npm run build`: lint and production-build frontend.
- `python -m experiments.fine_tuning.basic_qlora` (or other `python -m experiments.<track>.<module>`): run experiments from repo root.
- `uv run pytest`: run Python tests when present.

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` for functions/modules, `UPPER_SNAKE_CASE` for constants.
- TypeScript/React: 2-space indentation, `PascalCase` components (`AttentionPage.tsx`), `camelCase` functions/vars, hooks prefixed with `use` (for example, `useWebSocket.ts`).
- Keep experiment files aligned with `experiments/EXPERIMENT_TEMPLATE.py` (clear hypothesis/method/results structure).
- Run `npm run lint` (frontend) and `uv run ruff check .` (Python) before opening a PR.

## Testing Guidelines
- Framework: `pytest` (configured as a dev dependency). Current automated coverage is limited.
- Add focused tests for new logic, especially API routers/services and reusable experiment utilities.
- For training pipeline changes, include at least one lightweight smoke run (for example, `python -m experiments.pretraining.train --config nano --corpus tiny --epochs 1`) and summarize results in the PR.
- For dashboard changes, include manual verification steps and screenshots.

## Commit & Pull Request Guidelines
- Follow the existing history style: short, imperative commit subjects (`Add ...`, `Fix ...`, `Improve ...`); optional milestone prefixes like `Phase N:`.
- Avoid vague messages (`stuff`, `123`); each commit should describe a single intent.
- PRs should include: what changed, why, commands/tests run, related issue/question, and screenshots for UI updates.

## Security & Configuration Tips
- Do not commit `.env` files, model checkpoints, or downloaded corpora; `.gitignore` already excludes common large/runtime artifacts.
- Keep secrets and local paths out of experiment configs and docs.
