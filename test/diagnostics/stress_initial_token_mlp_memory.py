"""Exercise the failed batch and largest raw-pixel batch without deleting tokens."""
import json
from pathlib import Path

import torch

from analysis.fig01b_hidden_prediction.initial_token_mlp_probe import Bank, PathHook, teacher, runtime, inputs_for, run_model, TRAIN
from analysis.fig01a_hidden_channels.visual_channel_native_cache import load_rows, dump_json

root = Path('artifacts/diagnostics/initial_token_mlp_qwen_20260916')
runtime()
processor, model = teacher()
print('PROCESSOR', processor.image_processor.size, flush=True)
saved = torch.load(root / 'attempt1_oom_step637/best.pt', map_location='cpu', mmap=True, weights_only=False)
bank = Bank(saved['stats'], saved['init']).cuda()
bank.load_state_dict(saved['bank'])
del saved
optimizer = torch.optim.AdamW(bank.parameters(), lr=3e-4, betas=(.9, .95), fused=True)
rows = load_rows(root / 'train.jsonl')
areas = json.loads(Path(str(TRAIN) + '.pixel_areas.json').read_text())['areas']
candidates = []
for step in range(2000):
    for rank in range(8):
        start = (step * 32 + rank * 4) % len(rows)
        pixels = sum(areas[rows[(start + j) % len(rows)]['source_row']] for j in range(4))
        candidates.append((pixels, step, rank))
_, max_step, max_rank = max(candidates)
cases = [(636, 2), (max_step, max_rank)]
hook = PathHook(model)
results = []
for step, rank in cases:
    torch.cuda.reset_peak_memory_stats()
    start = (step * 32 + rank * 4) % len(rows)
    batch = [rows[(start + j) % len(rows)] for j in range(4)]
    inp = inputs_for(processor, batch, model.device)
    hook.begin(inp, 'capture', cache_native=False)
    run_model(model, inp, hook)
    assert not hook.native
    optimizer.zero_grad(set_to_none=True)
    loss, _ = bank(hook.initial, hook.targets, hook.sizes)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.)
    optimizer.step()
    item = dict(step=step+1, original_rank=rank, visual_tokens=hook.sizes, loss=float(loss),
                gradient_norm=float(grad), peak_memory_bytes=torch.cuda.max_memory_allocated())
    assert torch.isfinite(loss) and torch.isfinite(grad)
    results.append(item)
    print('STRESS', item, flush=True)
    del loss
dump_json(root / 'memory_stress_validation.json', results)
