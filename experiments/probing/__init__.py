# Track C: Representation Probing
#
# Tools for extracting and analyzing internal activations (residual stream).
#
# Key concepts:
# - Residual stream: The "information highway" that flows through the model
# - Each layer reads from and writes to the residual stream
# - Activations capture what the model "knows" at each point
#
# See docs/concepts/representations-self-assessment.md for theory.

# Lazy imports to allow verification without torch
def extract_activations(*args, **kwargs):
    """Extract activations from model. Requires torch."""
    from .extract import extract_activations as _extract
    return _extract(*args, **kwargs)

def ActivationExtractor(*args, **kwargs):
    """Activation extraction class. Requires torch."""
    from .extract import ActivationExtractor as _Extractor
    return _Extractor(*args, **kwargs)

__all__ = ["extract_activations", "ActivationExtractor"]
