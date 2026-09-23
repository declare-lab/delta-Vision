"""Reproducible question sampling shared by all methods in an evaluation."""
import random


def sample_evaluation_rows(rows, *, limit=1000, seed=44):
    """Uniform sampling without replacement, then original-order evaluation.

    Sorting sampled indices does not change which questions were randomly drawn.
    Keep all questions when fewer than ``limit`` are available.
    """
    if limit <= 0:
        raise ValueError('Evaluation sample limit must be positive')
    indices = sorted(random.Random(seed).sample(range(len(rows)), min(limit, len(rows))))
    return [rows[index] for index in indices], indices
