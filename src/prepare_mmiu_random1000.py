"""Freeze an unfiltered random MMIU subset before any model is evaluated."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/diagnostics/mmiu_random1000_seed42_all_methods_20260914'
SOURCE = ROOT/'artifacts/diagnostics/embedding_adapter_corrected_20260914/mmiu_context_and_question_v2.jsonl'


def main():
    rows = [json.loads(line) for line in SOURCE.open() if line.strip()]
    indices = random.Random(42).sample(range(len(rows)),1000)
    chosen = [dict(rows[i], source_manifest_index=i) for i in indices]
    assert len(set(indices))==1000
    invalid = []
    for sample_index,r in enumerate(chosen):
        assert r['source_question'].strip().casefold() in r['question'].casefold()
        assert r['images']
        r['annotation_valid'] = r['answer'] in [chr(65+j) for j in range(len(r['choices']))]
        if not r['annotation_valid']:
            invalid.append(dict(sample_index=sample_index,source_index=r['index'],gold=r['answer'],choices=r['choices']))
        assert all((ROOT/'data/benchmarks/mmiu'/p).is_file() for p in r['images'])
    OUT.mkdir(parents=True,exist_ok=False)
    manifest=OUT/'mmiu_random1000.jsonl'
    manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen))
    (OUT/'manifest_map.json').write_text(json.dumps({'mmiu':str(manifest)},indent=2)+'\n')
    sampling=dict(strategy='uniform random without replacement over source rows; random draw order retained; no task balancing or score filtering',
        seed=42,source=str(SOURCE),source_count=len(rows),sample_count=len(chosen),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),source_indices=indices,
        task_counts=dict(sorted(Counter(r['task'] for r in chosen).items())),
        image_counts=dict(sorted(Counter(len(r['images']) for r in chosen).items())),
        invalid_annotations=invalid,invalid_annotation_policy='Keep fixed sample and original gold; report all-1000 and valid-annotation subset separately')
    (OUT/'sampling.json').write_text(json.dumps(sampling,indent=2)+'\n')
    print(json.dumps({k:v for k,v in sampling.items() if k!='source_indices'},indent=2))


if __name__=='__main__':
    main()
