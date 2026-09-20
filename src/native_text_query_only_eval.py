"""Qwen3-VL ablation: remove visual query-side attention, keep text queries native.

Functional definition:
- In every language layer prefill, attention output rows corresponding to visual
  tokens are set to zero after o_proj.
- Visual token states therefore pass through residual + per-token FFN only.
- Text token rows are unchanged: text queries can attend to visual K/V and text
  K/V under the model's native causal mask.
- Decode is native text-only cache decode. No adapters, no training.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

from src.initial_token_mlp_probe import runtime, teacher
from src.visual_cross_token_ablation import prepare
from src.data import QwenBenchmarkDataset
from src.benchmarks import get_benchmark_spec, score_prediction
from src.eval_benchmarks import generate_teacher_qwen

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('TEXT_QUERY_ONLY_OUT', str(ROOT / 'artifacts/diagnostics/native_text_query_only_ffn_20260917')))
DATA_ROOT = ROOT / 'artifacts/diagnostics/channel_native_cache_20260916'
DATASETS = ('realworldqa', 'mmstar', 'sqa')


class TextQueryOnlyFFN:
    def __init__(self, model):
        self.enabled = False
        self.positions: torch.Tensor | None = None
        self.changed: list[int] = []
        self.handles = []
        for li, layer in enumerate(model.model.language_model.layers):
            self.handles.append(layer.self_attn.o_proj.register_forward_hook(self._zero_visual_attention(li)))

    def reset(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.changed = []

    def _zero_visual_attention(self, li: int):
        def hook(module, args, output):
            if not self.enabled:
                return
            if output.shape[1] == 1:
                return
            if self.positions is None or len(self.positions) == 0:
                raise RuntimeError('visual positions are not set')
            y = output.clone()
            y[:, self.positions] = 0
            self.changed.append(li)
            return y
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _score(name: str, item: dict[str, Any], answer_text: str) -> dict[str, Any]:
    return score_prediction(
        metric=get_benchmark_spec(name).metric,
        prediction_text=answer_text,
        answer=item.get('answer'),
        answers=item.get('answers'),
        choices=item.get('choices'),
        question=item.get('row', {}).get('question'),
    )


def worker(shard: int, limit: int) -> None:
    runtime()
    processor, model = teacher()  # FA2 + DeepStack disabled
    control = TextQueryOnlyFFN(model)
    depth = len(model.model.language_model.layers)
    row_path = OUT / f'rows{shard}.jsonl'
    done_path = OUT / f'done{shard}.json'
    with torch.inference_mode(), row_path.open('w', buffering=1) as out:
        for name in DATASETS:
            manifest_path = DATA_ROOT / f'{name}_eval.jsonl'
            ds = QwenBenchmarkDataset(str(manifest_path), processor, name)
            digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            n = min(len(ds), int(limit)) if limit else len(ds)
            for i in range(shard, n, 8):
                item = ds[i]
                inputs = prepare(item, model.device)
                positions = (inputs['input_ids'][0] == model.config.image_token_id).nonzero().flatten()
                assert len(positions) > 0, (name, i)
                assert torch.equal(positions, torch.arange(int(positions[0]), int(positions[-1]) + 1, device=positions.device)), (name, i, positions[:5], positions[-5:])
                control.reset(positions)
                for mode in ('native', 'text_query_only_visual_ffn'):
                    control.enabled = (mode != 'native')
                    control.changed = []
                    if hasattr(model.model, 'rope_deltas'):
                        model.model.rope_deltas = None
                    _, answer = generate_teacher_qwen(
                        model,
                        processor,
                        **inputs,
                        max_new_tokens=get_benchmark_spec(name).max_new_tokens,
                    )
                    expected = [] if mode == 'native' else list(range(depth))
                    assert control.changed == expected, (name, i, mode, control.changed[:5], control.changed[-5:], len(control.changed), depth)
                    score = _score(name, item, answer)
                    out.write(json.dumps({
                        'dataset': name,
                        'sample': i,
                        'mode': mode,
                        'answer': answer,
                        **score,
                        'manifest_sha256': digest,
                        'visual_tokens': int(len(positions)),
                        'changed_layers': control.changed,
                    }, ensure_ascii=False) + '\n')
                if (i - shard) % (8 * 20) == 0:
                    print(f'shard={shard} dataset={name} sample={i}/{n}', flush=True)
    done_path.write_text(json.dumps({'complete': True, 'shard': shard, 'time': time.time()}))


def summarize(root: Path, limit: int) -> dict[str, Any]:
    rows = []
    for shard in range(8):
        path = root / f'rows{shard}.jsonl'
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    summary: dict[str, Any] = {}
    for name in DATASETS:
        manifest_path = DATA_ROOT / f'{name}_eval.jsonl'
        total = sum(1 for _ in manifest_path.open())
        n = min(total, int(limit)) if limit else total
        summary[name] = {}
        for mode in ('native', 'text_query_only_visual_ffn'):
            selected = [r for r in rows if r['dataset'] == name and r['mode'] == mode]
            samples = {int(r['sample']) for r in selected}
            assert len(selected) == n and samples == set(range(n)), (name, mode, len(selected), n, sorted(set(range(n)) - samples)[:10])
            summary[name][mode] = {
                'samples': n,
                'accuracy_pct': 100.0 * sum(float(r['score']) for r in selected) / n,
                'invalid_pct': 100.0 * sum(1 for r in selected if r.get('invalid')) / n,
            }
    return summary


def launch(limit: int) -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    protocol = {
        'experiment': 'native_text_query_only_visual_ffn',
        'model': 'Qwen3-VL-4B-Instruct',
        'attention': 'flash_attention_2',
        'deepstack': 'off',
        'datasets': list(DATASETS),
        'samples': 'all rows unless limit is set',
        'definition': 'All LM layers: visual-token attention output rows are zeroed after o_proj during prefill. Visual states keep residual and FFN only. Text-token attention rows are untouched and can attend to visual/text K/V with the native causal mask. Decode q_len=1 is native text-only cache decode.',
        'modes': ['native', 'text_query_only_visual_ffn'],
        'world_size': 8,
        'max_new_tokens': {name: get_benchmark_spec(name).max_new_tokens for name in DATASETS},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    status = {'state': 'running', 'started': time.time(), 'limit': limit}
    (OUT / 'status.json').write_text(json.dumps(status, indent=2))
    jobs = []
    try:
        for shard in range(8):
            log = (OUT / f'gpu{shard}.log').open('w')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS='4')
            p = subprocess.Popen(
                [sys.executable, '-u', '-m', 'src.native_text_query_only_eval', 'worker', str(shard), str(limit)],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            jobs.append((p, log))
        while True:
            states = [p.poll() for p, _ in jobs]
            if any(state not in (None, 0) for state in states):
                raise RuntimeError(f'worker failed states={states}')
            if all(state == 0 for state in states):
                break
            time.sleep(5)
        summary = summarize(OUT, limit)
        (OUT / 'results.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        status.update(state='complete', finished=time.time(), results=summary)
    except BaseException as exc:
        status.update(state='failed', error=repr(exc), finished=time.time())
        raise
    finally:
        (OUT / 'status.json').write_text(json.dumps(status, indent=2, ensure_ascii=False))
        for p, log in jobs:
            if p.poll() is None:
                p.terminate()
            log.close()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        worker(int(sys.argv[2]), int(sys.argv[3]))
    else:
        limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        launch(limit)
