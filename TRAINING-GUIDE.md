# GPT Pretraining Guide

This guide walks through running pretraining experiments in the LLM Learning Lab. The goal is to build intuition about training dynamics, not to train production models.

## Quick Start

```bash
# Start the dashboard
./run-dashboard.sh

# Open http://localhost:5173 and navigate to Pretraining
```

Or run training directly from Python:

```bash
source .venv/bin/activate
python -m experiments.pretraining.train --config nano --corpus verdict --epochs 3
```

## Model Configurations

| Config | Parameters | Context | Training Time* | Use Case |
|--------|------------|---------|----------------|----------|
| nano   | ~10M       | 256     | Minutes        | Quick iteration, testing ideas |
| small  | ~50M       | 512     | 30-60 min      | Moderate experiments |
| medium | ~124M      | 1024    | Hours          | Serious experiments |

*On RTX 5060 Ti 16GB with "verdict" corpus

**Recommendation**: Start with `nano` to verify everything works, then scale up.

## Included Corpora

### verdict (default)
- **Source**: "The Verdict" by Edith Wharton (Project Gutenberg)
- **Size**: ~3K tokens
- **Good for**: Quick testing, verifying training works
- **Limitation**: Too small for meaningful learning; model will memorize

### tiny
- **Source**: A few sentences about machine learning
- **Size**: ~100 tokens
- **Good for**: Smoke testing the pipeline
- **Limitation**: Essentially useless for learning

## Alternative Datasets

The included corpora are intentionally small for quick iteration. For more meaningful experiments, consider these alternatives:

### TinyStories (Recommended for Learning)

**What**: 2.1M synthetic short stories written by GPT-3.5/4, designed for training small language models.

**Why it's great for learning**:
- Specifically designed to show what small models (10M-100M params) can learn
- Clean, consistent grammar with limited vocabulary
- Stories follow coherent narratives, so you can evaluate quality subjectively
- The paper "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" shows nano-scale models can produce grammatical text

**How to use**:
```bash
# Download from HuggingFace
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('roneneldan/TinyStories'); ds['train'].to_pandas()['text'].str.cat(sep='\n\n')[:5000000]" > experiments/pretraining/corpus/tinystories.txt
```

Then add to `experiments/pretraining/data.py`:
```python
CORPUS_REGISTRY = {
    'verdict': 'verdict.txt',
    'tiny': 'tiny.txt',
    'tinystories': 'tinystories.txt',  # Add this
}
```

### WikiText-2

**What**: 2M tokens from Wikipedia's "Good" and "Featured" articles.

**Why it's useful**:
- Standard benchmark for language modeling
- Well-structured expository text
- Good variety of topics and sentence structures

**How to use**:
```python
from datasets import load_dataset
ds = load_dataset('wikitext', 'wikitext-2-raw-v1')
with open('experiments/pretraining/corpus/wikitext2.txt', 'w') as f:
    f.write('\n'.join(ds['train']['text']))
```

### Shakespeare

**What**: Complete works of Shakespeare (~1M tokens)

**Why it's interesting**:
- Distinctive patterns (meter, archaic vocabulary)
- Easy to evaluate: does generated text "sound Shakespearean"?
- Public domain, well-studied

**How to use**:
```bash
curl -o experiments/pretraining/corpus/shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### Custom Text Files

Any plain text file works. Place it in `experiments/pretraining/corpus/` and either:
1. Add it to `CORPUS_REGISTRY` in `data.py`
2. Or pass the full path directly: `--corpus /path/to/your/text.txt`

Good sources for public domain text:
- [Project Gutenberg](https://www.gutenberg.org/) - Classic literature
- [Internet Archive](https://archive.org/details/texts) - Various texts
- [Common Crawl](https://commoncrawl.org/) - Web text (requires filtering)

## Training Run Walkthrough

### Run 1: Verify the Pipeline (5 minutes)

```bash
# Via dashboard
./run-dashboard.sh
# Set: config=nano, corpus=verdict, epochs=3

# Or via CLI
python -m experiments.pretraining.train --config nano --corpus verdict --epochs 3
```

**What to observe**:
- Loss should decrease rapidly (overfitting to small corpus)
- Final loss should be very low (<0.5) since model memorizes the text
- Generated text should reproduce fragments from the training data

### Run 2: Actual Learning with TinyStories (30-60 minutes)

```bash
# First, download TinyStories (see above)
python -m experiments.pretraining.train --config nano --corpus tinystories --epochs 1
```

**What to observe**:
- Loss decreases more gradually
- Loss won't go as low (model can't memorize 2M stories)
- Generated text should be grammatical but novel (not memorized)
- Watch for:
  - Coherent sentence structure
  - Proper punctuation
  - Story-like flow ("Once upon a time...")

### Run 3: Compare Model Sizes

Train the same corpus with different configs:

```bash
python -m experiments.pretraining.train --config nano --corpus tinystories --epochs 1
python -m experiments.pretraining.train --config small --corpus tinystories --epochs 1
```

**What to observe**:
- Larger model has lower training loss
- Does lower loss = better generations? (subjectively evaluate)
- Training time differences

## Metrics to Watch

### Training Loss
- **High and not decreasing**: Learning rate too low, or bug in code
- **Decreasing smoothly**: Normal training
- **Decreasing then jumping up**: Learning rate too high
- **Very low (<0.1)**: Probably overfitting/memorizing

### Validation Loss
- **Tracks training loss**: Good generalization
- **Diverges from training loss**: Overfitting starting
- **Validation loss increases while train decreases**: Clear overfitting

### Tokens/Second
- Measures throughput
- Useful for comparing batch sizes and optimizations

### Sample Generations
- The most informative metric for understanding what the model learned
- Watch how they evolve during training:
  - Early: Random gibberish
  - Middle: Word-like tokens, broken grammar
  - Late: Coherent phrases, sentences

## Common Issues

### Out of Memory
- Reduce batch size
- Use smaller context length
- Use nano config instead of small/medium

### Training Loss Stuck
- Learning rate may be too low
- Try warmup_steps=100 with higher base LR
- Check that data is loading correctly

### Generated Text is Gibberish
- Model needs more training
- Corpus may be too diverse for model size
- Try lower temperature for generation

### Training is Slow
- Reduce context length (biggest impact)
- Reduce batch size (trades memory for speed)
- Ensure GPU is being used (check nvidia-smi)

## Learning Exercises

1. **Memorization vs Generalization**: Train nano on verdict until loss is ~0.1. Generate text. Is it memorized or novel? Now train on TinyStories. What's different?

2. **Learning Rate Sensitivity**: Train the same config/corpus with LR 1e-4, 3e-4, 1e-3, 3e-3. Plot loss curves. Where does training become unstable?

3. **Context Length Impact**: Train nano with context_length=64, 128, 256. How does it affect what the model learns? Can you see differences in generated text coherence?

4. **Scaling Laws**: Train nano, small, medium on the same corpus for the same number of tokens. Plot loss vs parameters. Does it match expected scaling behavior?

## Next Steps

After exploring pretraining:
- **Attention Track**: Visualize what attention patterns emerge in your trained models
- **Probing Track**: Examine activations to understand what representations form
- **Fine-tuning Track**: Take a pretrained checkpoint and fine-tune on a specific task
