# Deslop Experiment: Reverse Distillation for Academic Writing

## Inspiration

[Unslopper-30B-A3B](https://huggingface.co/N8Programs/Unslopper-30B-A3B-bf16) — a LoRA fine-tune that learns to rewrite AI-generated text into more human-like prose. The key insight is the **reverse distillation** data pipeline: take human-written text, iteratively degrade it with an LLM to amplify AI-typical patterns, then train a model on (degraded → original) pairs.

We're adapting this approach for **academic/research writing** rather than literary fiction.

## Decisions Made

- **AI-ification API**: Claude Haiku — cost-effective, keeps things in one ecosystem
- **Dataset scale**: Moderate — targeting ~5,000 passages
- **Fine-tuning model**: Qwen2.5-7B-Instruct with QLoRA (4-bit), using existing Unsloth infrastructure
- **Hardware**: RTX 5060 Ti 16GB — sufficient for QLoRA on 7B at 4-bit
- **Task framing**: "Deslop" academic writing — given an AI-slopified version of a research paper passage, recover the original precise academic prose

## Why Academic Writing?

Academic prose has characteristics that contrast sharply with LLM defaults:

- **Precise hedging** ("our results suggest" vs. "this clearly shows")
- **Information density** (no filler, every sentence advances the argument)
- **Citation conventions** and attribution patterns
- **Section-specific voice** (abstract vs. methods vs. discussion)
- **Technical terminology** used precisely, not as decoration
- **Structural conventions** (topic sentences, logical connectives, signposting)

LLMs tend to flatten all of this into generic explainer prose. The "slop signal" should be strong and well-defined, making it a good learning vehicle for understanding what LoRA actually captures about style.

## Open Questions

### 1. Paper Source Selection (highest leverage decision)

**Options under consideration:**

| Source | Pros | Cons |
|--------|------|------|
| arXiv (LaTeX source) | Clean text extraction, CC BY license, massive scale | LaTeX parsing has edge cases, math-heavy passages may not "sloppify" well |
| arXiv (abstract only) | Very clean, consistent structure, no parsing needed | Short passages, limited stylistic range |
| Semantic Scholar ORC | Pre-extracted text | May have extraction artifacts |
| PubMed Central OA | High-quality biomedical writing | Narrower domain |
| ACL Anthology | Well-structured NLP papers | Small corpus, very narrow domain |

**Sub-questions:**
- Should we restrict to a specific domain (e.g., ML/NLP papers) or go broad?
  - Narrower = more coherent style signal, but risk of overfitting to subfield jargon
  - Broader = more generalizable, but "good academic writing" varies across fields
- How do we filter for **well-written** papers? Citation count is a proxy but imperfect. Manual curation of a seed set?
- Do we want full paper text or just certain sections?
  - Abstracts are clean and self-contained but stylistically narrow
  - Intro/discussion sections have the richest prose
  - Methods sections are highly structured — different kind of writing
  - Could include section labels as context for the model

### 2. Passage Extraction Strategy

- **Passage length**: 200-500 words seems right (matches Unslopper), but academic paragraphs can be longer
- **Section awareness**: Should passages carry metadata about which section they came from? This would let the model learn that "deslop" means different things for an abstract vs. a methods section
- **Filtering**: Need to strip LaTeX artifacts, references sections, acknowledgments, figure captions, tables, math-heavy passages that won't translate well to a text-only style transfer task
- **Quality control**: Some automated filtering (e.g., minimum paragraph length, coherence heuristics), plus possibly a small manually-reviewed seed set

### 3. AI-ification Pipeline Design

The Unslopper used **10 iterative rewrites** per passage, progressively amplifying AI patterns. Questions:

- **How many iterations?** 10 may be overkill — diminishing returns likely set in. Could experiment with 5 vs. 10 and compare the resulting "slop level"
- **Prompt design for sloppification**: Needs careful tuning. For academic text, the prompt should encourage specific degradation patterns:
  - Replacing hedged claims with overconfident statements
  - Adding filler and fluff
  - Genericizing technical terminology
  - Breaking information density with unnecessary elaboration
  - Losing citation/attribution patterns
- **Which iteration to use as training input?** The Unslopper used only the final (most degraded) version. Could also use intermediate iterations for a curriculum-style dataset
- **Batch processing / rate limiting**: At 5,000 passages x N iterations, need to handle API rate limits, resumability on failure, and cost tracking

### 4. Training Configuration

Starting point based on existing infrastructure + Unslopper's config:

| Parameter | Unslopper | Our Starting Point | Notes |
|-----------|-----------|-------------------|-------|
| LoRA rank | 8 | 8-16 | Could sweep this — connects to existing lora_rank experiments |
| Alpha | 20 | 16-32 | |
| Learning rate | 1e-4 | 2e-4 | Existing QLoRA setup uses 2e-4 |
| Epochs | 1 (1000 iter) | 1-3 | With 5K examples, 1 epoch = 5K steps |
| Batch size | 1 | 4 | We have grad accumulation infrastructure |
| Max seq length | 6144 | 1024-2048 | 16GB VRAM limits this at 4-bit |
| Optimizer | Adam | AdamW 8-bit | Existing setup uses adamw_8bit |

### 5. Evaluation Methodology

Three dimensions to measure:

**A. "Humanness" / AI detection evasion**
- Unslopper used Pangram API (paid)
- Alternatives: Binoculars (open source, perplexity-based), GPTZero API, or custom perplexity analysis
- Could compute perplexity under a reference model as a proxy — human academic text has characteristic perplexity distributions distinct from AI text

**B. Writing quality preservation**
- LLM-as-judge scoring (Claude Sonnet or Opus evaluating coherence, precision, style)
- Specific rubric for academic writing: technical accuracy, appropriate hedging, information density, structural coherence
- Compare: original paper passage vs. sloppified version vs. desloped version

**C. Control comparison**
- Same prompts through base Qwen2.5-7B without fine-tuning
- Measures what the LoRA adapter specifically learned vs. what the base model can already do with prompting alone

**Open eval questions:**
- Do we need a held-out test set from different papers/authors than training?
- How do we handle the subjective nature of "good academic writing"?
- Can we design automated metrics that correlate with human judgment of academic prose quality?

## Existing Infrastructure to Build On

- **PG-19 corpus**: Already downloaded, but we're pivoting to academic papers — new download/extraction pipeline needed
- **QLoRA training**: `experiments/fine_tuning/basic_qlora.py` and `api/routers/fine_tuning.py` — adapter injection, SFTTrainer, dashboard monitoring all ready
- **Dashboard**: Real-time training metrics via WebSocket — can reuse for monitoring the fine-tuning run
- **Experiment template**: `experiments/EXPERIMENT_TEMPLATE.py` — hypothesis/methodology/results structure

## Next Steps (once source questions are resolved)

1. Build paper extraction pipeline (download, parse, chunk into passages)
2. Build sloppification pipeline (Haiku API calls, iterative rewriting, storage)
3. Format as training dataset (chat template pairs)
4. Run QLoRA fine-tuning with dashboard monitoring
5. Evaluate on held-out set across all three dimensions
6. Analyze what the LoRA weights learned (connects to existing probing/attention experiments)
