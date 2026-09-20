"""Observe token-layer sums and KV bytes for six pruning implementations."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'artifacts/diagnostics/qwen_baseline_layer_tokens_20260918'
METHODS = ['base', 'fastv', 'dart', 'visionzip', 'divprune', 'zoo', 'sparsevlm']


def worker(method):
    import torch
    from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item, _qwen_speed_resources
    from src.data import QwenBenchmarkDataset
    from src.qwen_deepstack import disable_qwen_deepstack
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model, processor = load_baseline_model(method, '/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct',
                                           torch.bfloat16, torch.device('cuda:0'), .2, 'flash_attention_2')
    disable_qwen_deepstack(model)
    model.eval()
    item = QwenBenchmarkDataset(str(ROOT/'data/benchmarks/mmstar/mmstar_val.jsonl'), processor, 'mmstar', max_samples=1)[0]
    inputs = _qwen_inputs_from_item(item, torch.device('cuda:0'))
    mask = item['mm_token_type_ids'].ne(0) & item['attention_mask'].bool()
    positions = mask.nonzero().flatten()
    nv = len(positions)
    nt = int(item['attention_mask'].sum()) - nv
    lm = model.model.language_model
    lengths = []
    handles = [block.register_forward_pre_hook(lambda module,args,kw: lengths.append(int(kw.get('hidden_states',args[0] if args else None).shape[1])),
                with_kwargs=True) for block in lm.layers]
    records = []
    for retention in ([1.] if method == 'base' else [.05,.2]):
        configure_baseline(model, method, retention, int(positions[0]), nv)
        lengths.clear()
        lm._pruning_audit = []
        model.model.rope_deltas = None
        torch.manual_seed(42)
        with torch.inference_mode():
            output = model(**inputs, use_cache=True, return_dict=True, logits_to_keep=1)
        assert len(lengths) == len(lm.layers), lengths
        kv_lengths = [int(layer.keys.shape[-2]) for layer in output.past_key_values.layers]
        assert lengths == kv_lengths, (lengths, kv_lengths)
        actual_bytes = sum(t.numel()*t.element_size() for layer in output.past_key_values.layers for t in (layer.keys,layer.values))
        resource = _qwen_speed_resources(model,item,method,retention,torch.bfloat16)
        records.append(dict(method=method, nominal_retention=retention, original_visual=nv, text=nt,
                            layers=len(lengths), layer_total_tokens=list(lengths), layer_visual_tokens=[n-nt for n in lengths],
                            sum_all_tokens=sum(lengths), sum_visual_tokens=sum(n-nt for n in lengths),
                            all_token_ratio=sum(lengths)/(len(lengths)*(nv+nt)),
                            visual_token_ratio=sum(n-nt for n in lengths)/(len(lengths)*nv),
                            actual_kv_mb=actual_bytes/1024**2, estimated_kv_mb=resource['kv_cache_mb'],
                            estimate_matches_actual=abs(actual_bytes/1024**2-resource['kv_cache_mb'])<1e-8,
                            pruning_audit=lm._pruning_audit))
        del output
    for handle in handles:
        handle.remove()
    (OUT/f'{method}.json').write_text(json.dumps(records,indent=2))
    print(method,'complete',flush=True)


def launch():
    OUT.mkdir(parents=True,exist_ok=False)
    jobs = []
    try:
        for gpu,method in enumerate(METHODS):
            log=(OUT/f'{method}.log').open('w')
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),method],cwd=ROOT,
                               env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
            jobs.append((p,log,method))
        while any(p.poll() is None for p,_,_ in jobs):
            time.sleep(3)
        failed=[method for p,_,method in jobs if p.returncode != 0]
        results=[row for _,_,method in jobs if (OUT/f'{method}.json').exists() for row in json.loads((OUT/f'{method}.json').read_text())]
        (OUT/'summary.json').write_text(json.dumps(dict(failed=failed,results=results),indent=2))
        assert not failed, failed
    finally:
        for p,log,_ in jobs:
            if p.poll() is None:
                p.terminate()
            log.close()


if __name__ == '__main__':
    worker(sys.argv[1]) if len(sys.argv)>1 else launch()
