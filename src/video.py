"""Video sampling and fixed Video-MME model inputs."""


# Full-window video sampling with real timestamps and benchmark clip boundaries.
from pathlib import Path
from math import sqrt
import av
import numpy as np
from PIL import Image


def limit_image_pixels(image, maximum):
    if maximum and image.width * image.height > maximum:
        scale = sqrt(maximum / (image.width * image.height))
        resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
        image.close()
        return resized
    return image


def sample_benchmark_video(path, row, count=8, max_pixels=262144):
    path = Path(path)
    frames, times = [], []
    if path.is_dir():
        paths = sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        # MVBench's episodic_reasoning source is tvqa/video_fps3_hq_segment.
        fps = row.get("fps") or (3 if row.get("task") == "episodic_reasoning" else None)
        if not paths or fps is None:
            raise ValueError(f"Frame directory requires images and a known fps: {path}")
        fps = float(fps)
        duration = len(paths) / fps
        begin, end = window(row, duration, fps)
        for target in np.linspace(begin, end, count):
            idx = min(len(paths) - 1, max(0, round(target * fps)))
            with Image.open(paths[idx]) as source:
                image = limit_image_pixels(source.convert("RGB"), max_pixels)
            frames.append(image)
            times.append(idx / fps)
    else:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            fps = float(stream.average_rate or stream.base_rate or 30)
            origin = float((stream.start_time or 0) * stream.time_base)
            duration = (float(stream.duration * stream.time_base) if stream.duration is not None
                        else float(container.duration / av.time_base) if container.duration else None)
            if not duration or duration <= 0:
                raise ValueError(f"Video has no usable duration: {path}")
            begin, end = window(row, duration, fps)
            for target in np.linspace(begin, end, count):
                container.seek(int((origin + target) / float(stream.time_base)), stream=stream, backward=True)
                chosen = None
                for frame in container.decode(stream):
                    chosen = frame
                    if frame.time is not None and frame.time - origin >= target - 0.5 / fps:
                        break
                if chosen is None:
                    if not frames:
                        raise ValueError(f"No decodable frames: {path}")
                    frames.append(frames[-1].copy())
                    times.append(times[-1])
                else:
                    frames.append(limit_image_pixels(chosen.to_image().convert("RGB"), max_pixels))
                    times.append(max(0.0, float(chosen.time or origin) - origin))
    metadata = {"fps": 1000.0, "duration": duration,
                "total_num_frames": max(1, round(duration * 1000)),
                "frames_indices": [round(t * 1000) for t in times]}
    return frames, metadata


def window(row, duration, fps):
    begin = max(0.0, float(row.get("start") or 0))
    end = min(float(row["end"]) if row.get("end") is not None else duration,
              max(0.0, duration - 1 / fps))
    if begin > end:
        raise ValueError(f"Invalid video window start={begin}, end={end}, duration={duration}")
    return begin, end


# Video-MME inputs matching the saved 999-row full_timestamp_v1 protocol.
import hashlib
import json
import os
import re

import torch
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
