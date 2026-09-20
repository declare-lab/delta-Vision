"""Use the repaired annotations in every multi-image evaluation entry point."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def benchmark_manifest(benchmark):
    if benchmark == 'mmiu':
        path = ROOT / 'artifacts/diagnostics/embedding_adapter_corrected_20260914/mmiu_context_and_question_v2.jsonl'
        if not path.is_file():
            raise FileNotFoundError(f'Repaired MMIU annotations are required; refusing question-omitting legacy input: {path}')
        return path
    return ROOT / f'data/benchmarks/{benchmark}/test.jsonl'
