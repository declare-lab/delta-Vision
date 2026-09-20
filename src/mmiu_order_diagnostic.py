"""Paired cyclic-image-order diagnostic on all 132 BLINK first-1000 rows."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/diagnostics/adapter_nodeepstack_mediafirst_20260913'
OUTPUT = ROOT / 'artifacts/diagnostics/mmiu_order_nodeepstack_20260913'


def worker(shard):
    import src
    src.__path__.insert(0, str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src.data import QwenBenchmarkDataset
    from src import model as ref
    from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
    from src.benchmarks import score_prediction
    torch.set_num_threads(4)
    model, processor = load_baseline_model('base', '/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct',
        torch.bfloat16, 'cuda:0', 1., 'sdpa')
    checkpoint = ROOT/'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt'
    adapter, meta = ref.load_qwen_embedding_adapter_checkpoint(checkpoint, model.model.language_model,
        torch.device('cuda'), torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.eval().requires_grad_(False)
    adapter.eval().requires_grad_(False)
    def off(module, args, kwargs):
        return args, dict(kwargs, deepstack_visual_embeds=None)
    def reject(*args, **kwargs):
        raise AssertionError('DeepStack executed')
    model.model.language_model.register_forward_pre_hook(off, with_kwargs=True)
    model.model.language_model._deepstack_process = reject
    dataset = QwenBenchmarkDataset(str(ROOT/'data/benchmarks/mmiu/test.jsonl'), processor, 'mmiu',
        max_samples=1000, prompt_layout='media_first_v1')
    indices = [i for i, r in enumerate(dataset.rows) if r['task'] == 'forensic_detection_blink']
    assert len(indices) == 132
    before = {(r['index'],r['method']):r for p in SOURCE.glob('rows_*.jsonl')
              for line in p.read_text().splitlines() if (r:=json.loads(line))['benchmark']=='mmiu'}
    with torch.inference_mode(), (OUTPUT/f'rows_{shard}.jsonl').open('w',buffering=1) as out:
        for j in range(shard, len(indices), 8):
            index=indices[j]; row=dataset.rows[index]
            assert len(row['images'])==4 and row['answer'] in 'ABCD'
            # Same pixels as files, only [0,1,2,3] -> [1,2,3,0]. The textual
            # options remain first/second/third/fourth; gold rotates accordingly.
            dataset.rows[index]=dict(row,images=row['images'][1:]+row['images'][:1],
                answer=chr(65+(ord(row['answer'])-65-1)%4))
            item=dataset[index]; initial_inputs=_qwen_inputs_from_item(item,torch.device('cuda'))
            for method in ['base','static_kl']:
                inputs=dict(initial_inputs);generated=[];eos=model.generation_config.eos_token_id
                eos=eos if isinstance(eos,list) else [eos]
                for step in range(8):
                    model.model.rope_deltas=None
                    if method=='base':logits=model(**inputs,use_cache=False,logits_to_keep=1).logits
                    else:logits=ref.qwen_embedding_adapter_logits(model,adapter,inputs,logits_to_keep=1)[0]
                    token=int(logits[0,-1].argmax());generated.append(token)
                    text=processor.tokenizer.decode(generated,skip_special_tokens=True).strip()
                    if token in eos or text in list('ABCD'):break
                    new=torch.tensor([[token]],device='cuda',dtype=inputs['input_ids'].dtype)
                    inputs['input_ids']=torch.cat([inputs['input_ids'],new],1)
                    inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
                    inputs['mm_token_type_ids']=torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(new)],1)
                result=score_prediction(metric='mmiu',prediction_text=text,answer=item['answer'],choices=item['choices'])
                old=before[index,method];pred=old['prediction']
                expected=chr(65+(ord(pred)-65-1)%4) if pred in list('ABCD') else None
                out.write(json.dumps(dict(index=index,method=method,original_gold=row['answer'],
                    rotated_gold=item['answer'],original_prediction=pred,original_score=old['score'],
                    rotated_prediction=result['prediction'],rotated_score=result['score'],
                    follows_same_image=expected is not None and result['prediction']==expected,
                    expected_rotated_prediction=expected,deepstack_enabled=False))+'\n')
            print('DONE', index,flush=True)


def run():
    assert json.loads((SOURCE/'status.json').read_text())['state']=='complete'
    OUTPUT.mkdir(parents=True,exist_ok=True)
    jobs=[];logs=[]
    for shard in range(8):
        log=(OUTPUT/f'worker{shard}.log').open('w');logs.append(log)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.mmiu_order_diagnostic',str(shard)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for log in logs:log.close()
    assert not any(codes),codes
    rows=[json.loads(l) for p in OUTPUT.glob('rows_*.jsonl') for l in p.read_text().splitlines()]
    result={}
    from collections import Counter
    for method in ['base','static_kl']:
        rs=[r for r in rows if r['method']==method]
        assert len(rs)==len({r['index'] for r in rs})==132
        result[method]=dict(samples=len(rs),original_accuracy=100*sum(r['original_score'] for r in rs)/len(rs),
            rotated_accuracy=100*sum(r['rotated_score'] for r in rs)/len(rs),
            same_image_consistency=100*sum(r['follows_same_image'] for r in rs)/len(rs),
            original_predictions=dict(Counter(r['original_prediction'] for r in rs)),
            rotated_predictions=dict(Counter(r['rotated_prediction'] for r in rs)))
    (OUTPUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
