"""Full-window video sampling with real timestamps and benchmark clip boundaries."""
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
