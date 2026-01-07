# LLM Learning Lab Implementation Review (PRD-LLM-LEARNING-LAB)

## Findings (ordered by severity)

### Critical
- Training scripts disable Unsloth patching and pass `eval_strategy` into `TrainingArguments`, which is not a valid parameter in stock Transformers. This will raise `TypeError` and block runs unless patching is enabled. Fix by using `evaluation_strategy` or removing `UNSLOTH_DISABLE_TRAINER_PATCHING=1`. Refs: `experiments/fine_tuning/basic_qlora.py:230`, `experiments/fine_tuning/loss_curve_analysis.py:356`, `experiments/learning_rate/experiment.py:196`, `experiments/lora_rank/experiment.py:203`, `qwen-finetune/scripts/test_finetune.py:80`.

### High
- Activation extraction advertises `post_attn` and downstream analyses depend on it, but `ActivationExtractor.capture` never records it. As a result, attention-vs-FFN contribution analyses either zero out or skip layers, invalidating Track C and paper reproduction results. Implement a post-attn hook (e.g., capture attention module output plus residual) or change the analysis to match what is actually captured. Refs: `experiments/probing/extract.py:104`, `experiments/probing/extract.py:208`, `experiments/probing/run_analysis.py:145`, `experiments/paper_reproduction/bayesian_geometry/experiment.py:117`.
- The attention comparison experiment does not compare anything: it loads and deletes the base model, comments out base extraction, and never calls the diff/visualization helpers. The output JSON contains only fine-tuned tokens. Implement sequential base/fine-tuned extraction (possibly saving tensors to disk) and run `compute_attention_diff` and `visualize_comparison`. Refs: `experiments/attention/compare_experiment.py:240`, `experiments/attention/compare_experiment.py:260`.
- Fine-tuning experiments explicitly avoid saving adapters, but attention comparison tries to load them from `outputs/.../adapter`. When no adapter is found it falls back to an untrained LoRA, so any comparison is meaningless. Save adapters (or call `model.save_pretrained`) in the training scripts or change the comparison workflow. Refs: `experiments/attention/compare_experiment.py:94`, `experiments/fine_tuning/basic_qlora.py:230`, `experiments/fine_tuning/loss_curve_analysis.py:356`, `experiments/learning_rate/experiment.py:196`, `experiments/lora_rank/experiment.py:203`, `qwen-finetune/scripts/test_finetune.py:80`.

### Medium
- Attention extraction defaults to capturing all layers and heads with full sequence length, which is O(L^2) memory per head and can easily blow RAM for 512-1024 tokens. Consider requiring explicit `layers`, adding a max token guard, or emitting a warning. Refs: `experiments/attention/extract.py:149`, `experiments/attention/extract.py:224`, `experiments/attention/extract.py:261`.
- Visualization utilities call `.numpy()` directly on tensors. This fails for CUDA tensors and makes the helpers brittle when used outside the current CPU-only extraction flow. Convert via `.detach().cpu().numpy()` for safety. Refs: `experiments/attention/visualize.py:81`, `experiments/attention/visualize.py:252`.

### Low
- Documentation mismatches implementation: the learning-rate README uses a non-existent module path and GETTING-STARTED lists output directories that do not match actual `output_dir` values, which will confuse users following the guide. Refs: `experiments/learning_rate/README.md:31`, `GETTING-STARTED.md:119`.
- Catastrophic forgetting evaluation uses sampling with substring matching, which introduces variance and false positives/negatives. For more reliable comparisons, disable sampling and/or use a stricter scoring harness. Refs: `experiments/forgetting/experiment.py:214`.

## Open Questions / Assumptions
- Is `UNSLOTH_DISABLE_TRAINER_PATCHING=1` intentional, or should Unsloth’s patched args remain enabled?
- Should `post_attn` reflect the attention module output before residual addition, or the post-residual state? This choice affects contribution math.
- Do you want attention comparisons to run sequentially with on-disk caching (lowest VRAM), or should the tool insist on a large VRAM setup?
- Should output directory names be standardized across experiments and docs, or should docs mirror current `output_dir` values?

## Suggested Fixes (high level)
- Replace `eval_strategy` with `evaluation_strategy` across training scripts (or remove the patching disable flag) to unblock training.
- Implement post-attention capture in `ActivationExtractor` and update analyses to use it.
- Save LoRA adapters during fine-tuning and update `compare_experiment` to load them reliably.
- Wire `compare_experiment` to perform base extraction + diff metrics + visualization, using disk caching if needed.
- Add guards or defaults to prevent attention extraction OOMs.
- Harden visualization conversions for GPU tensors and align docs with actual commands/paths.

## Suggested Tests
- Unit test for `ActivationExtractor` that asserts `pre_attn`, `post_attn`, and `post_ffn` are captured and that `get_layer_diff` works.
- Unit test for `extract_attention` that checks returned tensor shapes and layer/head counts for a tiny model.
- Smoke test that `TrainingArguments` initialization succeeds for each experiment config (no unexpected kwargs).
