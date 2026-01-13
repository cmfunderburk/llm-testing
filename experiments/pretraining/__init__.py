"""
LLM Pretraining Lab

A hands-on platform for understanding LLM pretraining from scratch,
featuring a GPT implementation with real-time visualization.

This is a LEARNING PROJECT - the goal is mental model formation,
not producing a capable language model.

Reference: Raschka, "Build a Large Language Model (From Scratch)"
See: docs/book-chapters/text/04-implementing-gpt-model.txt
     docs/book-chapters/text/05-pretraining-on-unlabeled-data.txt

Usage:
    # Training (CLI)
    python -m experiments.pretraining.train --config nano --epochs 10

    # Start API server
    python -m experiments.pretraining.api.main

    # Frontend (from experiments/pretraining/frontend/)
    npm run dev
"""
