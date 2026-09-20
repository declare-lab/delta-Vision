"""Expand the existing Video-R1 subset without altering the old 20k manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent/'vision-kv-inject-attention-sink/scripts'))
import prepare_adapter_multimodal_subset as prep


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT/'data/train/video_expansion_20260915')
    parser.add_argument('--workers', type=int, default=112)
    opts = parser.parse_args()
    root = opts.root.resolve()
    old = ROOT/'data/train/multimodal_subset_20260911'
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root/'.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    original_digest = digest(old/'video/train.jsonl')
    original = prep.jsonl(old/'video/train.jsonl')
    original_ids = {r['id'] for r in original}
    assert len(original_ids) == len(original) == 20000
    # Hard links reuse validated bytes, with independent manifests/status files.
    # The downloader never rewrites existing media. No old file is removed.
    for directory in ('source/Video-R1-data', 'video/media', 'video/archive_indices'):
        for source in (old/directory).rglob('*'):
            if not source.is_file() or '.cache' in source.parts or source.suffix == '.part':
                continue
            dest = root/source.relative_to(old)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                os.link(source, dest)
    status = root/'video/video_status.json'
    if not status.exists():
        shutil.copyfile(old/'video/video_status.json', status)
    prep.dump(root/'plan.json', dict(dataset=prep.VIDEO, revision=prep.REVISIONS[prep.VIDEO],
        target_qa=64000, old_qa=20000, old_manifest_sha256=original_digest,
        max_distinct_qa_per_video=4, seed=44, workers=opts.workers,
        video_storage_ceiling_gib=220, free_disk_floor_gib=100,
        preserved_media='Hard-linked original validated media; original manifests unchanged',
        training_started=False))
    # Exact benchmark ID exclusions must use the active repository's datasets.
    prep.HERE = ROOT
    args = argparse.Namespace(root=root, count=64000, seed=44, workers=opts.workers,
        max_qa_per_group=4, video_gib=220, video_timeout=120, image_timeout=40)
    prep.video(args)
    result = prep.jsonl(root/'video/train.jsonl')
    assert len(result) == len({r['id'] for r in result}) == 64000
    assert original_ids <= {r['id'] for r in result}, 'Expansion must preserve all original 20k QA'
    assert digest(old/'video/train.jsonl') == original_digest
    prep.dump(root/'complete.json', dict(qa=64000, old_qa_preserved=20000,
        new_qa=44000, distinct_qa_per_video_cap=4, original_manifest_unchanged=True,
        manifest_sha256=digest(root/'video/train.jsonl'), training_started=False))
    prep.log('video_expansion_complete', qa=64000, old_preserved=20000)


if __name__ == '__main__':
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','TOKIO_WORKER_THREADS'):
        os.environ[name] = '1'
    main()
