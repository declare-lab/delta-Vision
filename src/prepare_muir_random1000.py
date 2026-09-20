"""Freeze the shared MuirBench subset before evaluating any method.

Sample source rows uniformly, without replacement, with Python Random(42).
Keep original IDs, image order, questions, choices and labels unchanged. Retain
random draw order; do not select by task, model predictions or annotation label.
The full source and earlier first-1000 evaluation outputs remain untouched.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/benchmarks/muirbench/test.jsonl'
OUT = ROOT / 'artifacts/diagnostics/muir_random1000_seed42_matched_20260914'


def main():
    rows = [json.loads(line) for line in SOURCE.open() if line.strip()]
    assert len(rows) == 2600
    indices = random.Random(42).sample(range(len(rows)), 1000)
    assert len(indices) == len(set(indices)) == 1000
    chosen = [dict(rows[i], source_manifest_index=i) for i in indices]
    invalid = []
    for sample_index, row in enumerate(chosen):
        assert row['question'].strip() and row['images'] and row['choices']
        row['annotation_valid'] = row['answer'] in [chr(65 + i) for i in range(len(row['choices']))]
        if not row['annotation_valid']:
            invalid.append(dict(sample_index=sample_index, source_index=row['index'],
                                gold=row['answer'], choices=row['choices']))
        for raw in row['images']:
            assert (ROOT / 'data/benchmarks/muirbench' / raw).is_file(), raw
    assert len({row['index'] for row in chosen}) == 1000
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = OUT / 'muirbench_random1000.jsonl'
    manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in chosen))
    (OUT / 'manifest_map.json').write_text(json.dumps({'muirbench': str(manifest)}, indent=2) + '\n')
    sampling = dict(
        strategy='uniform random without replacement over source rows; random draw order retained; no task balancing or score filtering',
        seed=42, source=str(SOURCE), source_count=len(rows), sample_count=len(chosen),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        source_indices=indices,
        task_counts=dict(sorted(Counter(row['task'] for row in chosen).items())),
        image_counts=dict(sorted(Counter(len(row['images']) for row in chosen).items())),
        invalid_annotations=invalid,
        invalid_annotation_policy='Keep fixed sample and original gold; report all-1000 and valid-annotation subset separately',
    )
    (OUT / 'sampling.json').write_text(json.dumps(sampling, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in sampling.items() if k != 'source_indices'}, indent=2))


if __name__ == '__main__':
    main()
