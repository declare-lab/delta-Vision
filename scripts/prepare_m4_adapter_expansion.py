"""Append a pinned, train-only M4 multi-image subset to existing adapter data.

Never modifies old manifests or launches training. Image bytes are fetched from
selected ZIP members, CRC checked and decoded; no full archive extraction.
"""
import argparse
import collections
import concurrent.futures
from datetime import timedelta
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'vision-kv-inject-attention-sink/scripts'))
import prepare_adapter_multimodal_subset as prep
import hf_xet
from huggingface_hub import get_hf_file_metadata, hf_hub_download
from PIL import Image

REPO = 'lmms-lab/M4-Instruct-Data'
REV = '9078d03bb7442cef5a71771a29a2af9ec0f63134'
EXCLUDE = {'dreamsim', 'ALFRED', 'FlintstonesSV', 'PororoSV', 'scannet_frames_25k',
           'nuscenes', 'nextqa', 'star'}
SESSION = GROUP = None


def xet_range(meta, start, end):
    global SESSION, GROUP
    if SESSION is None:
        cfg = hf_xet.XetConfig().with_config({
            'client.connect_timeout': timedelta(seconds=8),
            'client.read_timeout': timedelta(seconds=25), 'client.retry_max_attempts': 1,
            'client.retry_max_duration': timedelta(seconds=25),
            'client.ac_max_download_concurrency': 2, 'client.ac_initial_download_concurrency': 2,
            'reconstruction.min_prefetch_buffer': 1024**2,
            'reconstruction.download_buffer_perfile_size': 8*1024**2})
        SESSION = hf_xet.XetSession(config=cfg)
        GROUP = SESSION.new_download_stream_group(
            token_refresh_url=f'https://huggingface.co/api/datasets/{REPO}/xet-read-token/{REV}')
    if start == end:
        return b''
    info = hf_xet.XetFileInfo(hash=meta['xet_hash'], file_size=meta['archive_size'])
    try:
        data = b''.join(GROUP.download_stream(info, start=start, end=end))
    except Exception:
        # Some Xet CDN ranges intermittently fail. An independently signed,
        # cache-busted HTTP range reads exactly the same pinned ZIP bytes.
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{meta['archive']}?download=true&range_request={time.time_ns()}"
        data = prep.range_get(url, start, end)
    if len(data) != end-start:
        raise ValueError('Incomplete Xet byte range')
    return data


class XetReader(io.RawIOBase):
    def __init__(self, meta):
        self.meta, self.pos = meta, 0
    def seekable(self):
        return True
    def readable(self):
        return True
    def tell(self):
        return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.meta['archive_size']+offset
        return self.pos
    def read(self, size=-1):
        end = self.meta['archive_size'] if size < 0 else min(self.meta['archive_size'], self.pos+size)
        data = xet_range(self.meta, self.pos, end)
        self.pos = end
        return data


def index_archive(root, prefix):
    path = root / 'indices' / f'{prefix}.json'
    if path.exists():
        return json.loads(path.read_text())
    name = prefix+'.zip'
    meta = get_hf_file_metadata(f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{name}', token=False)
    assert meta.xet_file_data is not None
    base = dict(archive=name, archive_size=meta.size, xet_hash=meta.xet_file_data.file_hash)
    result = {}
    local = root/'source'/name
    prep.log('index_start', archive=name)
    if local.is_file():
        base['local_archive'] = str(local)
        archive = zipfile.ZipFile(local)
    else:
        try:
            archive = zipfile.ZipFile(XetReader(base))
        except Exception as exc:
            prep.log('range_index_fallback', archive=name, error=prep.error_text(exc),
                     action='Official whole-archive download; still extract selected members only')
            local = Path(hf_hub_download(REPO, name, repo_type='dataset', revision=REV, local_dir=root/'source'))
            base['local_archive'] = str(local)
            archive = zipfile.ZipFile(local)
    with archive as z:
        for item in z.infolist():
            if not item.is_dir():
                result[item.filename.removeprefix('./')] = dict(base, member=item.filename,
                    offset=item.header_offset, size=item.file_size, compressed_size=item.compress_size,
                    crc=item.CRC, compression=item.compress_type)
    prep.dump(path, result)
    prep.log('indexed', archive=name, members=len(result))
    return result


def image_job(root, source, entry):
    try:
        dest = root / 'images' / (hashlib.sha256(source.encode()).hexdigest()+PurePosixPath(source).suffix)
        if dest.exists():
            data = dest.read_bytes()
        else:
            import struct
            if entry.get('local_archive'):
                with Path(entry['local_archive']).open('rb') as handle:
                    handle.seek(entry['offset'])
                    header = handle.read(30)
                    fields = struct.unpack('<4s5H3I2H', header)
                    blob = header+handle.read(fields[-2]+fields[-1]+entry['compressed_size'])
            else:
                header = xet_range(entry, entry['offset'], entry['offset']+30)
                fields = struct.unpack('<4s5H3I2H', header)
                end = entry['offset']+30+fields[-2]+fields[-1]+entry['compressed_size']
                blob = xet_range(entry, entry['offset'], end)
            data = prep.decode_member(blob, entry)
        import zlib
        assert len(data) == entry['size'] and zlib.crc32(data) == entry['crc']
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            rgb = im.convert('RGB')
            pixel_sha = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            width, height = rgb.size
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            temporary = dest.with_suffix(dest.suffix+'.part')
            temporary.write_bytes(data)
            temporary.replace(dest)
        return dict(ok=True, path=str(dest), bytes=len(data), width=width, height=height,
                    sha256=hashlib.sha256(data).hexdigest(), pixel_sha256=pixel_sha, crc32=entry['crc'])
    except Exception as exc:
        return dict(ok=False, error=prep.error_text(exc))


def pixel_digest(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        return hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()


def benchmark_exclusion(root):
    cache = root/'benchmark_exclusion.json'
    manifests = [ROOT/f'data/benchmarks/{b}/test.jsonl' for b in ('muirbench','mmiu')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in manifests}
    if cache.exists():
        saved = json.loads(cache.read_text())
        assert saved['manifests'] == hashes
        return set(saved['pixels']), set(saved['questions'])
    paths, questions = set(), set()
    for p in manifests:
        for row in prep.jsonl(p):
            paths.update(str((p.parent/x).resolve()) for x in row.get('images', []))
            questions.add(re.sub(r'\s+', ' ', row['question']).strip().lower())
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        pixels = set(pool.map(pixel_digest, sorted(paths)))
    prep.dump(cache, dict(manifests=hashes, pixels=sorted(pixels), questions=sorted(questions),
                         scope='Exact decoded RGB and question overlap only; not perceptual deduplication'))
    return pixels, questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT/'data/train/m4_expansion_20260915')
    parser.add_argument('--old-root', type=Path, default=ROOT/'data/train/multimodal_subset_20260911')
    parser.add_argument('--count', type=int, default=44000)
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    root, old = args.root.resolve(), args.old_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root/'.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = hf_hub_download(REPO, 'm4_instruct_annotations.json', repo_type='dataset', revision=REV, local_dir=root/'source')
    rows = json.loads(Path(path).read_text())
    candidates, seen = [], set()
    skipped = collections.Counter()
    for index, row in enumerate(rows):
        images = row.get('image', [])
        if not isinstance(images, list) or not 2 <= len(images) <= 5:
            skipped['not_2_to_5_images'] += 1
            continue
        if row.get('metadata', {}).get('split') != 'train':
            skipped['not_explicit_train_split'] += 1
            continue
        prefixes = {p.split('/')[0] for p in images}
        if len(prefixes) != 1 or prefixes & EXCLUDE:
            skipped['video_3d_or_split_archive'] += 1
            continue
        conversation = row.get('conversations', [])
        if len(conversation) < 2 or conversation[0]['from'] != 'human' or conversation[1]['from'] != 'gpt':
            skipped['missing_first_pair'] += 1
            continue
        question, answer = conversation[0]['value'].strip(), conversation[1]['value'].strip()
        if question.count('<image>') != len(images) or not answer:
            skipped['placeholder_count_or_empty_answer'] += 1
            continue
        # Preserve each placeholder's explicit image identity when the existing
        # trainer moves all media before the question. Never drop A/B bindings.
        for i in range(len(images)):
            question = question.replace('<image>', f' Image {i+1} ', 1)
        question = question.strip()
        key = (tuple(images), question)
        if key in seen:
            skipped['duplicate_group_question'] += 1
            continue
        seen.add(key)
        candidates.append(dict(id=f'm4_{index}', source_row=index, source_dataset=next(iter(prefixes)),
            source_split='train', dataset=REPO, source_revision=REV, original_images=images,
            question=question, answer=answer, original_question=conversation[0]['value']))
    del rows
    random.Random(args.seed).shuffle(candidates)
    plan = dict(target_new_m4_qa=args.count, preserved_old_multi_qa=20000, seed=args.seed,
        revision=REV, annotation_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        eligible=len(candidates), candidates_by_source=dict(collections.Counter(r['source_dataset'] for r in candidates)),
        exclusions=dict(skipped), training_started=False,
        question_conversion='First human/assistant pair only; sequential <image> -> Image N; original order retained',
        old_manifest_sha256={k:hashlib.sha256((old/k/'train.jsonl').read_bytes()).hexdigest() for k in ('multi_image','video')})
    if (root/'plan.json').exists():
        assert json.loads((root/'plan.json').read_text()) == plan
    prep.dump(root/'plan.json', plan)
    prep.log('plan', **plan)
    if args.plan_only:
        return
    excluded_pixels, excluded_questions = benchmark_exclusion(root)
    # Bounded lookahead: request only the next candidate batch, not the entire
    # M4 release; enough spare candidates to replace invalid/overlapping media.
    state_path = root/'image_status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    seen_content_questions = {
        (tuple(row['image_sha256s']), re.sub(r'\s+', ' ', row['question']).strip().lower())
        for row in prep.jsonl(old/'multi_image/train.jsonl')}
    indices, selected, selected_areas = {}, [], []
    processed = 0
    def save(active=0):
        prep.room(root)
        prep.dump(state_path, state)
        prep.save_rows(root/'m4_train.partial.jsonl', selected)
        prep.dump(root/'status.json', dict(state='preparing', qa=len(selected), target=args.count,
            scanned=processed, active=active, images_ok=sum(v['ok'] for v in state.values()), skipped=dict(skipped)))
        prep.log('progress', qa=len(selected), target=args.count, scanned=processed, active=active)
    for start in range(0, len(candidates), 2048):
        if len(selected) == args.count:
            break
        batch = candidates[start:start+2048]
        requests = {}
        for row in batch:
            if row['source_dataset'] not in indices:
                indices[row['source_dataset']] = index_archive(root, row['source_dataset'])
            lookup = indices[row['source_dataset']]
            for source in row['original_images']:
                if source in state:
                    continue
                entry = lookup.get(source) or lookup.get(source.split('/',1)[1])
                if entry is None:
                    # Archives sometimes contain one common outer directory.
                    matches = [v for k,v in lookup.items() if k.endswith('/'+source)]
                    entry = matches[0] if len(matches) == 1 else None
                if entry is None:
                    state[source] = dict(ok=False, error='No unique exact ZIP member')
                else:
                    requests[source] = (root, source, entry)
        stream = prep.hard_deadline_map(image_job, requests.items(), workers=args.workers, timeout=60, heartbeat=save)
        try:
            for source, result in stream:
                state[source] = result
        finally:
            stream.close()
        for row in batch:
            processed += 1
            infos = [state[p] for p in row['original_images']]
            if not all(v['ok'] for v in infos):
                skipped['unavailable_media'] += 1
                continue
            if any(v['pixel_sha256'] in excluded_pixels for v in infos) or re.sub(r'\s+', ' ', row['question']).strip().lower() in excluded_questions:
                skipped['benchmark_overlap'] += 1
                continue
            content_question = (tuple(v['sha256'] for v in infos), re.sub(r'\s+', ' ', row['question']).strip().lower())
            if content_question in seen_content_questions:
                skipped['duplicate_content_question'] += 1
                continue
            seen_content_questions.add(content_question)
            selected.append(dict(row, images=[v['path'] for v in infos], image_sha256s=[v['sha256'] for v in infos],
                image_root=str(root), mixture_source='multi_image', train_max_image_pixels=1048576))
            selected_areas.append(sum(min(v['width']*v['height'],1048576) for v in infos))
            if len(selected) == args.count:
                break
        save()
    if len(selected) != args.count:
        raise RuntimeError(f'Only {len(selected)} valid M4 QA; no padding/repetition allowed')
    # Keep all old examples and their media roots byte-for-byte in the source;
    # build an independent new manifest, with no implicit training switch.
    originals = prep.jsonl(old/'mixed_train.jsonl')
    areas = json.loads((old/'mixed_train.jsonl.pixel_areas.json').read_text())['areas']
    pairs = [(r,a) for r,a in zip(originals,areas) if r['mixture_source'] in ('multi_image','video')]
    assert collections.Counter(r['mixture_source'] for r,a in pairs) == {'multi_image':20000,'video':20000}
    combined = [r for r,a in pairs]+selected
    all_areas = [a for r,a in pairs]+selected_areas
    prep.save_rows(root/'m4_train.jsonl', selected)
    prep.save_rows(root/'mixed_train.jsonl', combined)
    prep.dump(root/'mixed_train.jsonl.pixel_areas.json', dict(data=str(root/'mixed_train.jsonl'),count=len(combined),areas=all_areas))
    for kind, digest in plan['old_manifest_sha256'].items():
        assert hashlib.sha256((old/kind/'train.jsonl').read_bytes()).hexdigest() == digest
    prep.dump(root/'status.json',dict(state='complete',m4_qa=len(selected),
        combined_counts=dict(collections.Counter(r['mixture_source'] for r in combined)),
        sources=dict(collections.Counter(r['source_dataset'] for r in selected)),
        benchmark_exclusion='Exact decoded RGB and exact question only; no claim of perceptual deduplication',
        video_expansion_pending=True,training_started=False))
    prep.log('complete', manifest=str(root/'mixed_train.jsonl'), m4_qa=len(selected))


if __name__ == '__main__':
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','TOKIO_WORKER_THREADS'):
        os.environ[name] = '1'
    try:
        main()
    except Exception as exc:
        print(prep.error_text(exc), file=sys.stderr, flush=True)
        raise SystemExit(1) from None
