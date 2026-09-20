"""Wait for both expansions and assemble training data; never launch training."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent/'vision-kv-inject-attention-sink/scripts'))
from prepare_adapter_multimodal_subset import dump, jsonl, save_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m4-pid', type=int, required=True)
    parser.add_argument('--video-pid', type=int, required=True)
    args = parser.parse_args()
    m4 = ROOT/'data/train/m4_expansion_20260915'
    video = ROOT/'data/train/video_expansion_20260915'
    old = ROOT/'data/train/multimodal_subset_20260911'
    out = ROOT/'data/train/m4_multi64k_video64k_20260915'
    out.mkdir(parents=True, exist_ok=True)
    while True:
        ready = {'m4': (m4/'m4_train.jsonl').is_file() and (m4/'status.json').is_file()
                       and json.loads((m4/'status.json').read_text()).get('state') == 'complete',
                 'video': (video/'complete.json').is_file()}
        if all(ready.values()):
            break
        for name, pid in [('m4', args.m4_pid), ('video', args.video_pid)]:
            if not ready[name]:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    dump(out/'status.json', dict(state='blocked', reason=f'{name} preparation exited before completion'))
                    raise RuntimeError(f'{name} downloader exited; inspect its log before resuming')
        dump(out/'status.json', dict(state='waiting_for_data', ready=ready, training_started=False))
        time.sleep(20)
    old_rows = jsonl(old/'mixed_train.jsonl')
    old_areas = json.loads((old/'mixed_train.jsonl.pixel_areas.json').read_text())['areas']
    assert len(old_rows) == len(old_areas)
    pairs = [(r,a) for r,a in zip(old_rows,old_areas) if r['mixture_source']=='multi_image']
    assert len(pairs) == 20000
    rows, areas = [r for r,a in pairs], [a for r,a in pairs]
    new_multi = jsonl(m4/'m4_train.jsonl')
    assert len(new_multi) == 44000
    m4_images = {v['path']:v for v in json.loads((m4/'image_status.json').read_text()).values() if v['ok']}
    for row in new_multi:
        # Adjacent source placeholders must not become "Image 1Image 2".
        # The native formatter supplies actual images once, in source order.
        numbers = iter(range(1, len(row['images'])+1))
        question = re.sub(r'<image>', lambda _: f' Image {next(numbers)} ', row['original_question']).strip()
        assert '<image>' not in question
        row = dict(row, question=question)
        rows.append(row)
        areas.append(sum(min(m4_images[p]['width']*m4_images[p]['height'],1048576) for p in row['images']))
    video_state = json.loads((video/'video/video_status.json').read_text())
    new_video = jsonl(video/'video/train.jsonl')
    assert len(new_video) == 64000
    assert {r['id'] for r in jsonl(old/'video/train.jsonl')} <= {r['id'] for r in new_video}
    for row in new_video:
        info = video_state[row['source_path']]
        assert info['ok'] and row['videos'] == [info['path']]
        rows.append(dict(row, mixture_source='video', image_root=str(video), video_root=str(video),
                         num_frames=8, train_max_frame_pixels=262144))
        areas.append(8*min(info['width']*info['height'],262144))
    assert len(rows) == len({r['id'] for r in rows}) == 128000
    counts = Counter(r['mixture_source'] for r in rows)
    assert counts == {'multi_image':64000,'video':64000}
    media = set()
    for row in rows:
        for kind in ('images','videos'):
            for raw in row.get(kind,[]):
                path = Path(raw)
                if not path.is_absolute():
                    path = Path(row['video_root'] if kind=='videos' else row['image_root'])/path
                media.add(path)
    missing = [str(p) for p in media if not p.is_file() or p.stat().st_size == 0]
    assert not missing, missing[:10]
    sys.path.insert(0, str(ROOT.parent/'vision-kv-inject-attention-sink'))
    from src.multimodal_training import mixed_pixel_bucket_order
    order = mixed_pixel_bucket_order(rows, areas, ratios='multi_image:0.5,video:0.5',
        batch_size=32, steps=4000, bucket_size=512, seed=44)
    assert len(order) == len(set(order)) == 128000, 'No duplicated sample presentations'
    assert Counter(rows[i]['mixture_source'] for i in order) == counts
    manifest = out/'mixed_train.jsonl'
    save_rows(manifest, rows)
    dump(out/'mixed_train.jsonl.pixel_areas.json', dict(data=str(manifest),count=len(rows),areas=areas))
    dump(out/'integrity_audit.json', dict(passed=True, qa_counts=dict(counts),
        unique_media=len(media), missing_or_empty=missing, all_old_qa_preserved=True,
        media_verification='Original subset audit plus downloader CRC/decode checks; final file presence check',
        benchmark_exclusion='M4 explicit train split + exact RGB/question matching; Video-R1 benchmark IDs; not full perceptual deduplication',
        no_duplicate_sample_presentations_4000_steps=True,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))
    dump(out/'status.json',dict(state='ready',qa_counts=dict(counts),global_batch=32,total_steps=4000,
        steps_per_modality=2000,training_started=False,manifest=str(manifest)))
    print('DATA READY',manifest,flush=True)


if __name__ == '__main__':
    main()
