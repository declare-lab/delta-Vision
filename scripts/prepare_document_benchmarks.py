"""Download pinned public evaluation splits and freeze three 1000-question subsets."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random

from huggingface_hub import snapshot_download
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/benchmarks'
SOURCES = {
    'chartqa': ('lmms-lab/ChartQA', '9e63b7df1592a1c2158e735cc1725454aef0d6d9', 'data/test-*.parquet', 'test'),
    'docvqa': ('lmms-lab/DocVQA', '539088ef8a8ada01ac8e2e6d4e372586748a265e', 'DocVQA/validation-*.parquet', 'validation'),
    'infographicvqa': ('lmms-lab/DocVQA', '539088ef8a8ada01ac8e2e6d4e372586748a265e', 'InfographicVQA/validation-*.parquet', 'validation'),
}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def prepare(name):
    repo, revision, pattern, split = SOURCES[name]
    dest = DATA / name
    dest.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(repo, repo_type='dataset', revision=revision,
                              allow_patterns=[pattern, 'README.md'],
                              local_dir=dest/'source', max_workers=4)
    files = sorted(Path(local).glob(pattern))
    assert files, name
    rows = [row for file in files for row in pq.read_table(file).to_pylist()]
    rng = random.Random(42)
    if name == 'chartqa':
        groups = {kind: [i for i, row in enumerate(rows) if row['type'] == kind]
                  for kind in sorted({row['type'] for row in rows})}
        assert len(groups) == 2 and all(len(g) >= 500 for g in groups.values()), groups.keys()
        selected = sorted(i for group in groups.values() for i in rng.sample(group, 500))
    else:
        selected = sorted(rng.sample(range(len(rows)), 1000))
    images = dest/'images'
    images.mkdir(exist_ok=True)
    converted = []
    for index in selected:
        row = rows[index]
        answers = row.get('answers', row.get('answer'))
        answers = [answers] if isinstance(answers, (str, int, float)) else answers
        assert answers and all(isinstance(a, (str, int, float)) for a in answers), (name, index)
        answers = [str(a) for a in answers]
        blob = row['image']['bytes']
        digest = hashlib.sha256(blob).hexdigest()
        with Image.open(BytesIO(blob)) as im:
            extension = {'JPEG': 'jpg', 'PNG': 'png', 'TIFF': 'tif', 'WEBP': 'webp'}.get(im.format, 'img')
            width, height = im.size
            im.verify()
        image = images/f'{digest}.{extension}'
        if not image.exists():
            image.write_bytes(blob)
        assert sha(image) == digest
        converted.append(dict(index=index, question_id=row.get('questionId', index),
            question=row['question'], answer=answers[0], answers=answers,
            image=str(image.relative_to(dest)), image_root=str(dest),
            source_type=row.get('type'), source_split=split, source_repo=repo,
            source_revision=revision, image_sha256=digest, width=width, height=height))
    assert len(converted) == 1000 and len({r['index'] for r in converted}) == 1000
    manifest = dest/'eval1000_seed42.jsonl'
    text = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in converted)
    if manifest.exists():
        assert manifest.read_text() == text, 'Refusing to change frozen sample selection'
    else:
        manifest.write_text(text)
    metadata = dict(benchmark=name, repository=repo, revision=revision, split=split,
        source_samples=len(rows), samples=1000, seed=42,
        sampling='500 human + 500 augmented, random within strata' if name=='chartqa' else 'uniform random without replacement',
        source_files={str(p.relative_to(dest)): sha(p) for p in files},
        manifest=str(manifest), manifest_sha256=sha(manifest),
        unique_images=len({r['image'] for r in converted}),
        metric='chartqa_relaxed_accuracy' if name=='chartqa' else 'anls')
    (dest/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(prepare, SOURCES))
