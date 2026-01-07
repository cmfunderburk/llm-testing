# Track B: Attention Visualization
#
# Tools for extracting and visualizing attention patterns from transformer models.
#
# Key concepts:
# - Attention weights show how each token "attends" to other tokens
# - Multi-head attention: multiple parallel attention patterns
# - Patterns vary by layer (early = local, late = abstract?)
#
# See docs/concepts/attention-self-assessment.md for theory.

# Lazy imports to allow verification without torch installed
def extract_attention(*args, **kwargs):
    """Extract attention weights from a model. Requires torch."""
    from .extract import extract_attention as _extract
    return _extract(*args, **kwargs)

def AttentionExtractor(*args, **kwargs):
    """Attention extraction class. Requires torch."""
    from .extract import AttentionExtractor as _Extractor
    return _Extractor(*args, **kwargs)

__all__ = ["extract_attention", "AttentionExtractor"]
