"""Existing static adapter vs native visual layer inputs, RealWorldQA only."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from src.adapter_single_layer_similarity import (
    ROOT, DATA, CHECKPOINT, CHECKPOINT_SHA, INPUT_KEYS, sha, setup,
    vector_stats, aggregate_metrics, QwenBenchmarkDataset, _to_device_item, dump_json,
)


@torch.inference_mode()
def worker(root, shard):
    processor, model, adapter, previous_capture, meta = setup()
    previous_capture.close()
    adapter.eval()
    ds = QwenBenchmarkDataset(str(DATA/'realworldqa_eval.jsonl'), processor, 'realworldqa')
    assert len(ds) == 765
    state = dict(enabled=True)
    handles = []

    def capture(layer):
        def hook(module, args, kwargs):
            if not state['enabled']:
                return
            h = kwargs.get('hidden_states', args[0] if args else None)
            native = h.index_select(1, state['visual'])
            if layer == 0:
                state['initial'] = native.detach().clone()
            pred = adapter.visual_memory_for_layer(state['initial'], layer)
            assert layer not in state['metrics']
            state['metrics'][layer] = dict(adapter=vector_stats(pred[0], native[0]),
                                          identity=vector_stats(state['initial'][0], native[0]))
        return hook

    for layer, block in enumerate(model.model.language_model.layers):
        handles.append(block.register_forward_pre_hook(capture(layer), with_kwargs=True))
    checks = []
    started = time.time()
    with (root/f'rows_{shard}.jsonl').open('w', buffering=1) as log:
        for count, index in enumerate(range(shard, len(ds), 8)):
            item = _to_device_item(ds[index], model.device)
            inputs = {k:v for k,v in item.items() if k in INPUT_KEYS}
            state.update(metrics={}, visual=inputs['mm_token_type_ids'][0].eq(1).nonzero().flatten())
            assert len(state['visual']) > 0 and inputs['attention_mask'].bool().all()
            model.model.rope_deltas = None
            result = model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
            assert set(state['metrics']) == set(range(36))
            if count == 0:
                state['enabled'] = False
                model.model.rope_deltas = None
                plain = model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
                error = float((result.float()-plain.float()).abs().max())
                assert error == 0
                state['enabled'] = True
                checks.append(dict(index=index, capture_native_max_abs=error))
                del plain
            log.write(json.dumps(dict(index=index, visual_tokens=len(state['visual']),
                                      metrics=state['metrics']), allow_nan=False)+'\n')
            if count % 10 == 0:
                print('PROGRESS', shard, count+1, 'seconds', time.time()-started, flush=True)
            del result
    for h in handles:
        h.remove()
    dump_json(root/f'done_{shard}.json', dict(complete=True, checks=checks, step=meta['global_step'], seconds=time.time()-started))


def report(root):
    rows = [json.loads(s) for shard in range(8) for s in (root/f'rows_{shard}.jsonl').read_text().splitlines()]
    assert len(rows) == 765 and {r['index'] for r in rows} == set(range(765))
    results = {}
    table = []
    for layer in range(36):
        scores = {kind:aggregate_metrics([r['metrics'][str(layer)][kind] for r in rows])
                  for kind in ('adapter','identity')}
        results[str(layer)] = scores
        a, identity = scores['adapter']['macro'], scores['identity']['macro']
        table.append(dict(layer=layer,cosine=a['cosine'],mse=a['mse'],
                          identity_cosine=identity['cosine'],identity_mse=identity['mse']))
    # Exact overlap with the previous independent 13..22-layer measurement.
    old = ROOT/'artifacts/diagnostics/adapter_single_layer_20260916'
    reference = {}
    for shard in range(8):
        for line in (old/f'rows_{shard}.jsonl').open():
            r = json.loads(line)
            if r['dataset'] == 'realworldqa':
                reference[r['index'],r['layer']] = r['metrics']['hidden']
    overlap = 0
    maxdiff = dict(mse=0.,cosine=0.)
    for r in rows:
        for layer in range(13,23):
            a, b = r['metrics'][str(layer)]['adapter'], reference[r['index'],layer]
            for key in maxdiff:
                maxdiff[key] = max(maxdiff[key],abs(a[key]-b[key]))
            overlap += 1
    assert maxdiff['mse'] < 1e-8 and maxdiff['cosine'] < 1e-10, maxdiff
    dump_json(root/'results.json',dict(samples=765,layers=results,previous_overlap_pairs=overlap,previous_max_abs_difference=maxdiff))
    with (root/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    text=['# Existing embedding adapter: RealWorldQA, all 36 layers','',
          'Pixmo static KL step2000；Qwen3-VL-4B；FA2；BF16；DeepStack 关闭。',
          '预测 = E_i + adapter_l(E_i)，2560→128→2560；原生目标 = 对应层输入、RMSNorm 前的 visual hidden。',
          '同一原生前向采集所有层，预测不写回模型。MSE 为原始尺度 FP64 平方差；cosine 逐 visual token 计算，图像等权平均。',
          '层号为 0-based；765 条全部完成。Identity 对照直接使用 E_i。','',
          '| Layer | Adapter cosine ↑ | Adapter MSE ↓ | E_i cosine ↑ | E_i MSE ↓ |',
          '|---:|---:|---:|---:|---:|']
    for r in table:
        text.append(f"| {r['layer']} | {r['cosine']:.6f} | {r['mse']:.6f} | {r['identity_cosine']:.6f} | {r['identity_mse']:.6f} |")
    (root/'RESULTS.md').write_text('\n'.join(text)+'\n')


def launch(root):
    root.mkdir(parents=True,exist_ok=True)
    assert not (root/'status.json').exists(), 'Use a new output directory'
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    dump_json(root/'plan.json',dict(checkpoint=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,
        dataset=str(DATA/'realworldqa_eval.jsonl'),dataset_sha256=sha(DATA/'realworldqa_eval.jsonl'),
        samples=765,layers=list(range(36)),layer_target='input before RMSNorm, zero-based',
        prediction='existing checkpoint visual_memory_for_layer(E,l) = E + adapter_l(E)',
        attention='flash_attention_2',deepstack=False,source_sha256=sha(__file__),
        hidden_dimension=2560,adapter_rank=128,training=False,gpus=8))
    dump_json(root/'status.json',dict(state='running',pid=os.getpid()))
    children=[]
    try:
        for shard in range(8):
            log=(root/f'worker_{shard}.log').open('w')
            p=subprocess.Popen([sys.executable,'-u','-m','src.adapter_all_layers_hidden','worker','--output',str(root),'--shard',str(shard)],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false'),stdout=log,stderr=subprocess.STDOUT)
            children.append((p,log))
        codes=[p.wait() for p,_ in children]
        assert not any(codes), codes
        report(root)
        dump_json(root/'status.json',dict(state='complete',samples=765,layer_sample_pairs=765*36))
    except BaseException as exc:
        dump_json(root/'status.json',dict(state='failed',error=repr(exc)))
        raise
    finally:
        for p,log in children:
            if p.poll() is None:p.terminate()
            log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('launch','worker','report'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shard',type=int,default=0)
    args=parser.parse_args()
    if args.action=='worker':worker(args.output,args.shard)
    else:globals()[args.action](args.output)
