"""Inspect actual per-layer token counts; leave evaluation implementations unchanged."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from baselines.llava_hf_baselines import generate_llava_baseline, llava_dart_retained_image_token_indices
from src.data import LlavaBenchmarkDataset
from src.model import load_frozen_llava, _get_language_model

OUT = ROOT / 'artifacts/diagnostics/retention_layer_sum_audit_20260918'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.manual_seed(42)

# Run the recorded selector on controlled tensors: selected cardinality depends
# on the budget and number of pivots, not on the feature values in these cases.
layer = SimpleNamespace(input_layernorm=torch.nn.Identity(), self_attn=SimpleNamespace(k_proj=torch.nn.Identity(), head_dim=4))
lm = SimpleNamespace(norm=torch.nn.Identity())
selector_rows = []
for n in (576, 1152, 2880):
    h = torch.randn(1, n + 11, 8)
    for retention in (.05, .20):
        selected = llava_dart_retained_image_token_indices(layer, lm, h, image_start=3, image_len=n,
                    retention=retention, pivot_image_token=4, pivot_text_token=4)
        selector_rows.append(dict(visual_tokens=n, nominal_retention=retention, requested=round(n*retention),
                                  actual=len(selected), actual_visual_retention=len(selected)/n))

# The MoE selector receives B,H,S,D, unlike the LLaVA selector's B,S,H,D.
k = torch.arange(1*2*7*4).view(1,2,7,4)
wrong = k.reshape(1,7,-1)
correct = k.permute(0,2,1,3).reshape(1,7,-1)
layout = dict(input_shape=list(k.shape), direct_reshape_preserves_tokens=bool(torch.equal(wrong,correct)),
              direct_first_token=wrong[0,0].tolist(), proper_first_token=correct[0,0].tolist())

processor, model = load_frozen_llava('/lustre-data/leijingdi/code/delta-vision/models/llava-1.5-7b-hf',
                                     torch.bfloat16, 'cuda:0', 'flash_attention_2')
ds = LlavaBenchmarkDataset(str(ROOT/'data/benchmarks/mmstar/mmstar_val.jsonl'), processor, 'mmstar', max_samples=1)
item = ds[0]
inputs = {key:item[key].unsqueeze(0).to('cuda:0') for key in ('input_ids','attention_mask','pixel_values')}
image_id = model.config.image_token_index
nv = int((inputs['input_ids'] == image_id).sum())
nt = int(inputs['attention_mask'].sum()) - nv
lengths = []
handles = [block.register_forward_pre_hook(lambda module,args,kw: lengths.append(int(kw.get('hidden_states',args[0] if args else None).shape[1])),
                                           with_kwargs=True) for block in _get_language_model(model).layers]
records = []
for method,retention in [('base',1.),('dart',.05),('dart',.20),('divprune',.05),('divprune',.20)]:
    lengths.clear()
    answer = generate_llava_baseline(model, processor, **inputs, image_token_id=image_id, method=method,
                                     retention=retention, max_new_tokens=1)
    assert len(lengths) == 32, lengths
    visual = [n-nt for n in lengths]
    records.append(dict(method=method, nominal_retention=retention, answer_first_token=answer,
                        layers=32, original_visual_tokens=nv, text_tokens=nt, layer_total_tokens=list(lengths),
                        layer_visual_tokens=visual, sum_all_tokens=sum(lengths), sum_visual_tokens=sum(visual),
                        all_token_ratio=sum(lengths)/(32*(nv+nt)), visual_token_ratio=sum(visual)/(32*nv)))
for h in handles:
    h.remove()
result = dict(selector_cardinality=selector_rows, moe_k_layout=layout, actual_model_prefill=records,
              scope='Current 20% implementation; historical 5% five-model results unavailable. Not an accuracy rerun.',
              source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                             ['baselines/llava_hf_baselines.py','baselines/eval_baselines.py']})
(OUT/'audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
