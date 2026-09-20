"""Check document adapter cached generations against independent full-prefix forwards."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scripts.run_document_benchmarks import OUT, MODEL, CHECKPOINTS, BENCHES, manifest, dump
from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
from src.data import QwenBenchmarkDataset
from src.model import load_qwen_embedding_adapter_checkpoint
from src.eval_benchmarks import generate_adapter_qwen
from src.qwen_deepstack import disable_qwen_deepstack


def main():
    torch.set_num_threads(4)
    model,processor=load_baseline_model('base',MODEL,torch.bfloat16,torch.device('cuda:0'),1.,'flash_attention_2')
    model.eval().requires_grad_(False)
    disable_qwen_deepstack(model)
    model._adapter_attention_implementation='flash_attention_2'
    comparisons=[]
    with torch.inference_mode():
        for method,path in CHECKPOINTS.items():
            records=[json.loads(l) for l in (OUT/'smoke'/f'{method}_0.jsonl').read_text().splitlines()]
            adapter,meta=load_qwen_embedding_adapter_checkpoint(path,model.model.language_model,torch.device('cuda:0'),torch.bfloat16)
            assert not meta['missing'] and not meta['unexpected']
            adapter.eval().requires_grad_(False)
            for bench in BENCHES:
                dataset=QwenBenchmarkDataset(str(manifest(bench)),processor,bench)
                for old in [r for r in records if r['benchmark']==bench]:
                    item=dataset[old['index']]
                    inputs=_qwen_inputs_from_item(item,torch.device('cuda:0'))
                    torch.manual_seed(42+old['index'])
                    model.model.rope_deltas=None
                    _,text=generate_adapter_qwen(model,processor,adapter,**inputs,max_new_tokens=128)
                    comparison=dict(method=method,benchmark=bench,index=old['index'],
                        cached_text=old['prediction_text'],uncached_text=text.strip(),
                        same_text=old['prediction_text']==text.strip())
                    comparisons.append(comparison)
                    dump(OUT/'adapter_decode_parity.json',comparisons)
                    print(comparison,flush=True)
            del adapter
    assert all(c['same_text'] for c in comparisons), 'Cached and full-prefix answers differ; investigate before full evaluation'


if __name__=='__main__':
    main()
