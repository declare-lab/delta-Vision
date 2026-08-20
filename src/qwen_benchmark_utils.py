"""Compatibility shim for Qwen prefill benchmark helpers.

The implementation lives in src.benchmark_prefill so the prefill benchmark has a single Python file.
"""
from __future__ import annotations

from src.benchmark_prefill import *  # noqa: F401,F403
from src.benchmark_prefill import main as _benchmark_prefill_main


if __name__ == "__main__":
    _benchmark_prefill_main()
