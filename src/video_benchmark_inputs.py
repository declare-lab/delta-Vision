"""Video-MME inputs matching the saved 999-row full_timestamp_v1 protocol.

Cache only CPU processor inputs. Vision features and KV are recomputed on every
model request. The cache key matches the existing matched video evaluation.
"""
import hashlib
import json
import os
from pathlib import Path
import re

import torch
from src.benchmark_video_sampling import sample_benchmark_video
from src.benchmarks import build_benchmark_prompt


def video_benchmark_item(dataset, row, index):
    if row.get('images') or row.get('image'):
        raise ValueError('This video benchmark loader requires video-only rows')
    root = Path(row.get('video_root') or dataset.data_root)
    paths = [Path(p) if Path(p).is_absolute() else root / p
             for p in (row.get('videos') or [row['video']])]
    question = build_benchmark_prompt(row, dataset.spec, dataset.answer_instruction)
    frames = int(os.environ.get('QWEN_VIDEO_NUM_FRAMES', 8))
    sampling = os.environ.get('QWEN_VIDEO_SAMPLING', 'full_timestamp_v1')
    if sampling != 'full_timestamp_v1':
        raise ValueError('Video speed comparisons require full_timestamp_v1 sampling')
    cache_path = None
    if dataset.cache_dir is not None:
        stats = []
        for path in paths:
            stat = path.stat()
            stats.append(dict(path=str(path), size=int(stat.st_size), mtime_ns=int(stat.st_mtime_ns)))
        key = dict(benchmark=dataset.spec.name, prompt_layout='media_first_v1',
            video_sampling=sampling, video_num_frames=frames, video_start=row.get('start'),
            video_end=row.get('end'), processor=str(getattr(dataset.processor, 'name_or_path', '')),
            media=stats, question=str(row.get('problem') or question),
            answer_instruction=dataset.answer_instruction, index=row.get('index'))
        digest = hashlib.sha1(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cache_path = dataset.cache_dir / dataset.spec.name / f'{digest}.pt'
        if cache_path.exists():
            item = torch.load(cache_path, map_location='cpu', weights_only=False)['item']
            item.update(row=row, answer=row.get('answer'), answers=row.get('answers'),
                        choices=row.get('choices'), index=row.get('index', index))
            return item
    sampled = [sample_benchmark_video(path, row, frames) for path in paths]
    videos, metadata = [x[0] for x in sampled], [x[1] for x in sampled]
    refs = list(re.finditer(r'<\|(image|video)_(\d+)\|>', question))
    def replace(match):
        kind, number = match.group(1), int(match.group(2))
        if kind != 'video' or not 1 <= number <= len(videos):
            raise ValueError(f'Invalid video reference: {match.group(0)}')
        return f'Video {number}'
    text = re.sub(r'<\|(image|video)_(\d+)\|>', replace, question)
    content = []
    for number, video in enumerate(videos, 1):
        content.append(dict(type='video', video=video))
        if refs:
            content.append(dict(type='text', text=f'\n[End of Video {number}]\n'))
    content.append(dict(type='text', text=text))
    try:
        prompt = dataset.processor.apply_chat_template([dict(role='user', content=content)],
            tokenize=False, add_generation_prompt=True)
        inputs = dataset.processor(text=[prompt], videos=videos, video_metadata=metadata,
            do_sample_frames=False, return_tensors='pt', padding=True)
    finally:
        for video in videos:
            for frame in video:
                frame.close()
    if 'mm_token_type_ids' not in inputs:
        raise ValueError('Qwen processor did not return mm_token_type_ids')
    item = {key: inputs[key].squeeze(0) for key in ('input_ids', 'attention_mask', 'mm_token_type_ids')}
    item.update({key: inputs[key] for key in ('pixel_values_videos', 'video_grid_thw')})
    item.update(row=row, index=row.get('index', index), answer=row.get('answer'),
                answers=row.get('answers'), choices=row.get('choices'))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f'{cache_path.name}.tmp.{os.getpid()}')
        torch.save(dict(item=item), temporary)
        os.replace(temporary, cache_path)
    return item
