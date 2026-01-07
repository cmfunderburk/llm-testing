"""
Experiment: [Name]
==================

HYPOTHESIS
----------
State what you expect to happen and why.

Example:
> Training with learning rate X will cause Y because Z.

METHODOLOGY
-----------
Describe the experimental approach:
- Model/data configuration
- Variables being manipulated
- Metrics being captured
- Controls/baselines

WHAT WE'RE LEARNING
-------------------
What concepts does this experiment illuminate?
Link to relevant docs/concepts/ pages.

QUESTIONS TO ANSWER
-------------------
Specific questions this experiment addresses.
Link to QUESTIONS.md sections where applicable.
- Question 1? (see QUESTIONS.md#section)
- Question 2?

RESULTS
-------
[Fill after running]

Observations:
- Observation 1
- Observation 2

Data:
- Metric A: value
- Metric B: value

Visualizations:
- Link to plots or description of what was seen

LEARNINGS
---------
[Fill after running]

What did we learn?
- Learning 1
- Learning 2

Were hypotheses confirmed or refuted?
- Hypothesis about X: CONFIRMED/REFUTED because...

What questions were answered?
- Question 1: Answer + evidence

What new questions arose?
- New question 1
- New question 2

NEXT EXPERIMENTS
----------------
[Fill after running]

What experiments would logically follow?
- Experiment idea 1: to test X
- Experiment idea 2: to test Y
"""

# ============================================
# IMPORTS
# ============================================
# Standard library
import os

# Third party
# from unsloth import ...

# Local
# from experiments.utils import ...


# ============================================
# CONFIGURATION
# ============================================
# Hyperparameters and settings for this experiment
CONFIG = {
    "experiment_name": "template",
    "model_name": "unsloth/Qwen2.5-7B-Instruct",
    "seed": 42,
    # Add experiment-specific config
}


# ============================================
# EXPERIMENT CODE
# ============================================
def setup():
    """Initialize model, data, and experiment state."""
    pass


def run():
    """Execute the main experiment."""
    pass


def analyze():
    """Analyze results and generate visualizations."""
    pass


def run_experiment():
    """Main entry point."""
    print(f"Running experiment: {CONFIG['experiment_name']}")
    print("=" * 60)

    # 1. Setup
    print("\n[1/3] Setup...")
    setup()

    # 2. Run
    print("\n[2/3] Running experiment...")
    run()

    # 3. Analyze
    print("\n[3/3] Analyzing results...")
    analyze()

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print("\nREMINDER: Update the RESULTS and LEARNINGS sections above!")


if __name__ == "__main__":
    run_experiment()
