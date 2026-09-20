"""Read-only source/clip/frame audit for the currently evaluated video subsets."""
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/diagnostics/video_input_audit_20260914'


def run():
    import src
    src.__path__.insert(0, str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import av
    from datasets import load_dataset, DownloadConfig
    from src.benchmarks import build_benchmark_prompt, get_benchmark_spec, canonical_choice
    from src.benchmark_video_sampling import sample_benchmark_video, window
    raw_video = load_dataset('lmms-eval/Video-MME', split='test',
                             download_config=DownloadConfig(local_files_only=True))
    report = {}
    for bench in ('videomme', 'mvbench'):
        root = ROOT / f'data/benchmarks/{bench}'
        all_rows = [json.loads(l) for l in (root/'test.jsonl').open()]
        rows = all_rows[:1000]
        sources = {}
        for r in rows:
            prompt = build_benchmark_prompt(r, get_benchmark_spec(bench))
            if bench == 'videomme':
                raw = raw_video[r['index']]
                assert raw['videoID'] == r['videoID']
                assert raw['question'].strip() in prompt
                assert list(raw['options']) == r['choices']
                assert raw['answer'].strip() == r['answer']
                assert Path(r['videos'][0]).stem == raw['videoID']
            else:
                if r['task'] not in sources:
                    sources[r['task']] = json.loads((root/f"raw/json/{r['task']}.json").read_text())
                raw = sources[r['task']][r['task_index']]
                assert raw['question'].strip() in prompt
                assert list(raw['candidates']) == r['choices']
                assert raw['video'] == r['source_video']
                assert r['choices'][ord(r['answer'])-65] == raw['answer']
                for key in ('start','end'):
                    assert r.get(key) == raw.get(key)
            assert all(choice.strip() in prompt for choice in r['choices'])
            assert canonical_choice(r['answer'], r['choices']) == r['answer']
        def metadata(r):
            p = root/r['videos'][0]
            assert p.exists(), p
            if p.is_dir():
                files = sorted(f for f in p.iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
                fps = float(r.get('fps') or 3)
                duration = len(files)/fps
                numeric = [int(f.stem) for f in files]
                assert numeric == sorted(numeric), 'Lexical filename sorting changed temporal order'
                assert all(b-a == 1 for a,b in zip(numeric,numeric[1:])), 'Missing frame indices'
            else:
                with av.open(str(p)) as container:
                    stream = container.streams.video[0]
                    fps = float(stream.average_rate or stream.base_rate or 30)
                    duration = float(stream.duration*stream.time_base) if stream.duration is not None else container.duration/av.time_base
            begin,end = window(r,duration,fps)
            return dict(index=r['index'],duration=duration,fps=fps,start=begin,end=end,
                        annotation_end=r.get('end'),path=str(p))
        with ThreadPoolExecutor(max_workers=8) as pool:
            metas = list(pool.map(metadata, rows))
        groups = Counter(r.get('duration') if bench=='videomme' else r['task'] for r in rows)
        selected = {0,999}
        for group in groups:
            indices = [i for i,r in enumerate(rows) if (r.get('duration') if bench=='videomme' else r['task']) == group]
            selected.update((indices[0],indices[len(indices)//2],indices[-1]))
        selected.update(sorted(range(1000),key=lambda i:metas[i]['end']-metas[i]['start'])[:3])
        selected.update(sorted(range(1000),key=lambda i:metas[i]['end']-metas[i]['start'])[-3:])
        def frames(i):
            r, meta = rows[i],metas[i]
            images,video_meta = sample_benchmark_video(meta['path'],r,count=8)
            try:
                ts = [t/video_meta['fps'] for t in video_meta['frames_indices']]
                assert len(images)==len(ts)==8
                assert ts==sorted(ts), (i,ts)
                tolerance=1/meta['fps']+0.002
                assert ts[0]>=meta['start']-tolerance, (i,ts,meta)
                assert ts[-1]<=meta['end']+tolerance, (i,ts,meta)
                assert abs(ts[0]-meta['start'])<=tolerance, (i,ts,meta)
                assert abs(ts[-1]-meta['end'])<=tolerance, (i,ts,meta)
                return dict(index=i,timestamps=ts,start=meta['start'],end=meta['end'],
                            unique_timestamps=len(set(ts)),sizes=[im.size for im in images])
            finally:
                for im in images:im.close()
        with ThreadPoolExecutor(max_workers=8) as pool:
            frame_checks=list(pool.map(frames,sorted(selected)))
        clipped=[m for m in metas if m['annotation_end'] is not None and m['annotation_end']-m['duration']>1/m['fps']+0.01]
        def verify_end(meta):
            p=Path(meta['path'])
            if p.is_dir():
                n=sum(f.suffix.lower() in ('.jpg','.jpeg','.png','.webp') for f in p.iterdir())
                actual=(n-1)/meta['fps']
            else:
                with av.open(str(p)) as container:
                    stream=container.streams.video[0]
                    stream.codec_context.thread_count=1
                    origin=float((stream.start_time or 0)*stream.time_base)
                    n=0;actual=None
                    for frame in container.decode(stream):
                        n+=1
                        if frame.time is not None:actual=frame.time-origin
                    assert actual is not None
            return dict(index=meta['index'],decoded_or_directory_frames=n,actual_last_second=actual,
                        annotation_end=meta['annotation_end'],end_gap=meta['annotation_end']-actual)
        with ThreadPoolExecutor(max_workers=8) as pool:
            verified_ends=list(pool.map(verify_end,clipped))
        report[bench]=dict(samples=1000,total_manifest=len(all_rows),source_checks_passed=1000,
            media_checks_passed=len(metas),groups=dict(groups),frame_checks=frame_checks,
            annotation_ends_beyond_video=clipped,verified_media_ends=verified_ends)
        print(bench,'source/media PASS',len(metas),'sampled clips',len(frame_checks),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':run()
