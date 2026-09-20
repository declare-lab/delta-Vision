"""Prepare pinned official training splits and exclude held-out image matches."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random

from huggingface_hub import snapshot_download
import pyarrow.parquet as pq
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/train/document_qa_20260919'
SOURCES = {
    'chartqa': ('ahmed-masry/ChartQA', 'af8b6f5c08c95085271561c2a3f9d15f2b5a9031', 'data/train-*.parquet'),
    'docvqa': ('lmms-lab/DocVQA', '539088ef8a8ada01ac8e2e6d4e372586748a265e', 'DocVQA/train-*.parquet'),
    'infographicvqa': ('lmms-lab/DocVQA', '539088ef8a8ada01ac8e2e6d4e372586748a265e', 'InfographicVQA/train-*.parquet'),
}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def rgb_hash(blob):
    with Image.open(BytesIO(blob)) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        h = hashlib.sha256(str(im.size).encode())
        h.update(im.tobytes())
        return h.hexdigest()


def rows(files):
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=8):
            yield from batch.to_pylist()


def heldout():
    cache = OUT/'heldout_image_hashes.json'
    paths = sorted(p for b in SOURCES for p in (ROOT/'data/benchmarks'/b/'source').rglob('*.parquet'))
    fingerprints = {str(p): sha(p) for p in paths}
    if cache.exists():
        data = json.loads(cache.read_text())
        assert data['files'] == fingerprints
        return set(data['bytes']), set(data['rgb'])
    raw, rgb = set(), set()
    count = 0
    for row in rows(paths):
        blob = row['image']['bytes']
        digest = hashlib.sha256(blob).hexdigest()
        if digest not in raw:
            raw.add(digest)
            rgb.add(rgb_hash(blob))
        count += 1
    dump(cache, dict(files=fingerprints, rows=count, bytes=sorted(raw), rgb=sorted(rgb)))
    print('HELDOUT', count, 'questions', len(raw), 'unique encoded images', flush=True)
    return raw, rgb


def prepare(name, excluded_raw, excluded_rgb):
    repo, revision, pattern = SOURCES[name]
    dest = OUT/name
    dest.mkdir(exist_ok=True)
    if (dest/'provenance.json').exists():
        meta = json.loads((dest/'provenance.json').read_text())
        assert meta['repository'] == repo and meta['revision'] == revision
        assert sha(dest/'train.jsonl') == meta['manifest_sha256']
        assert meta['heldout_hashes_sha256'] == sha(OUT/'heldout_image_hashes.json')
        return [json.loads(l) for l in (dest/'train.jsonl').read_text().splitlines()]
    local = snapshot_download(repo, repo_type='dataset', revision=revision, allow_patterns=[pattern, 'README.md'],
        local_dir=dest/'source', max_workers=4)
    files = sorted(Path(local).glob(pattern))
    assert files
    images = dest/'images'
    images.mkdir(exist_ok=True)
    seen_images = {}
    counts = Counter()
    converted = []
    for index, row in enumerate(rows(files)):
        counts['source_rows'] += 1
        blob = row['image'] if isinstance(row['image'], bytes) else row['image']['bytes']
        digest = hashlib.sha256(blob).hexdigest()
        if digest not in seen_images:
            pixel_digest = rgb_hash(blob)
            seen_images[digest] = None
            if digest not in excluded_raw and pixel_digest not in excluded_rgb:
                with Image.open(BytesIO(blob)) as im:
                    width, height = im.size
                    ext = {'PNG':'png','JPEG':'jpg','TIFF':'tif','WEBP':'webp'}.get(im.format, 'img')
                    im.verify()
                path = images/f'{digest}.{ext}'
                if not path.exists():
                    path.write_bytes(blob)
                assert sha(path) == digest
                seen_images[digest] = (path, width, height, pixel_digest)
        meta = seen_images[digest]
        if meta is None:
            counts['heldout_image_rows_removed'] += 1
            continue
        answers = row.get('answers', row.get('answer', row.get('label')))
        if isinstance(answers, (str, int, float)):
            answers = [answers]
        answers = [str(a).strip() for a in (answers or []) if str(a).strip()]
        question = str(row.get('question', row.get('query', ''))).strip()
        if not answers or not question:
            counts['empty_qa_rows_removed'] += 1
            continue
        path, width, height, pixel_digest = meta
        converted.append(dict(image=str(path.resolve()), question=question,
            answer=answers[0], answers=answers, source_dataset=name, source_split='train', source_index=index,
            question_id=row.get('questionId', index), image_sha256=digest, rgb_sha256=pixel_digest,
            width=width, height=height))
    manifest = dest/'train.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in converted))
    counts['kept_rows'] = len(converted)
    counts['kept_images'] = len({r['image_sha256'] for r in converted})
    dump(dest/'provenance.json', dict(repository=repo, revision=revision, split='train', counts=dict(counts),
        files={str(p.relative_to(dest)):sha(p) for p in files}, manifest_sha256=sha(manifest),
        heldout_hashes_sha256=sha(OUT/'heldout_image_hashes.json')))
    print(name, dict(counts), flush=True)
    return converted


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    excluded_raw, excluded_rgb = heldout()
    with ThreadPoolExecutor(max_workers=3) as pool:
        groups = list(pool.map(lambda name: prepare(name, excluded_raw, excluded_rgb), SOURCES))
    combined = [r for group in groups for r in group]
    assert combined
    random.Random(44).shuffle(combined)
    manifest = OUT/'train.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in combined))
    dump(OUT/'pixel_areas.json', dict(data=str(manifest), count=len(combined), areas=[r['width']*r['height'] for r in combined]))
    dump(OUT/'manifest.json', dict(rows=len(combined), source_counts=dict(Counter(r['source_dataset'] for r in combined)),
        manifest_sha256=sha(manifest), seed=44, mixture='all retained training QA rows, natural proportions',
        heldout_policy='Exclude matching encoded bytes OR exact decoded RGB pixels against all locally downloaded held-out splits across all three datasets; no claim of perceptual near-duplicate removal.',
        sources={name:json.loads((OUT/name/'provenance.json').read_text()) for name in SOURCES}))
    print('COMPLETE', len(combined), manifest, flush=True)


if __name__ == '__main__':
    main()
