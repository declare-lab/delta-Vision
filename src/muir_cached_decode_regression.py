"""Old/new cache masks versus full recomputation on identical token prefixes."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'artifacts/diagnostics/muir_cached_decode_regression_20260914'


def worker(shard):
    import src
    src.__path__.insert(0, str(ROOT.parent/'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.embedding_adapter_corrected_eval import MODEL, CHECKPOINT
    from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model, processor = load_baseline_model('base', MODEL, torch.bfloat16, 'cuda:0', 1., 'sdpa')
    adapter, meta = ref.load_qwen_embedding_adapter_checkpoint(
        CHECKPOINT, model.model.language_model, torch.device('cuda'), torch.bfloat16)
    assert not meta['missing'] and not meta['unexpected']
    model.eval().requires_grad_(False)
    adapter.eval().requires_grad_(False)
    dataset = QwenBenchmarkDataset(str(ROOT/'data/benchmarks/muirbench/test.jsonl'),
        processor, 'muirbench', max_samples=1000, prompt_layout='media_first_v1')
    index = [440,450,461,471,482,492,503,513][shard]
    inputs = _qwen_inputs_from_item(dataset[index], torch.device('cuda'))
    fixed_mask = ref._qwen_decode_attention_mask
    def legacy_mask(cache, position_ids, current_text_mask=None):
        wrong_coordinates = dict(cache, next_text_positions=position_ids[0,:,0].view(-1,1))
        return fixed_mask(wrong_coordinates, position_ids, current_text_mask)
    def compare(reference, candidate):
        r, c = reference[0,-1].float(), candidate[0,-1].float()
        return dict(kl=float((r.softmax(-1)*(r.log_softmax(-1)-c.log_softmax(-1))).sum()),
            same_argmax=bool(r.argmax()==c.argmax()))
    with torch.inference_mode():
        hidden, pos = ref.build_qwen_initial_context(model, inputs)
        logits = ref.qwen_embedding_adapter_logits(model, adapter, inputs,
            initial_hidden=hidden, position_ids=pos, logits_to_keep=1)[0]
        prefix_logits, _, cache = ref.qwen_embedding_adapter_prefill_cache(model, adapter,
            inputs['input_ids'],inputs['attention_mask'],inputs['mm_token_type_ids'],hidden,pos)
        caches = {name:copy.deepcopy(cache) for name in ['legacy','fixed','hf_static']}
        ref.qwen_embedding_adapter_attach_hf_static_cache(model,caches['hf_static'],max_new_tokens=4)
        initial_physical_position = cache['next_text_positions'].clone()
        nvisual = cache['image_mask'].shape[1]
        result = dict(index=index, visual_tokens=nvisual, prefill=compare(logits,prefix_logits),
            first_decode_rope_coordinate=int(cache['next_position_ids'][0,0,0]),
            first_decode_sequence_position=int(initial_physical_position[0,0]),
            legacy_visible_visual_tokens=int(legacy_mask(cache,cache['next_position_ids'])[0,0,0,:nvisual].sum()),
            fixed_visible_visual_tokens=int(fixed_mask(cache,cache['next_position_ids'])[0,0,0,:nvisual].sum()),steps=[])
        for step in range(4):
            # The same reference token goes to every implementation, even after
            # EOS, solely to test the next-step logits rather than divergent text.
            token = logits[:,-1].argmax(-1).view(1,1)
            inputs['input_ids'] = torch.cat([inputs['input_ids'],token],1)
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(token)],1)
            hidden = torch.cat([hidden,model.model.get_input_embeddings()(token)],1)
            pos = torch.cat([pos,pos[:,:,-1:]+1],2)
            logits = ref.qwen_embedding_adapter_logits(model,adapter,inputs,
                initial_hidden=hidden,position_ids=pos,logits_to_keep=1)[0]
            scores = {}
            for name in caches:
                ref._qwen_decode_attention_mask = legacy_mask if name=='legacy' else fixed_mask
                try:
                    fn = ref.qwen_embedding_adapter_decode_step_hf_static if name=='hf_static' else ref.qwen_embedding_adapter_decode_step
                    candidate, caches[name] = fn(model,adapter,token,caches[name])
                finally:
                    ref._qwen_decode_attention_mask = fixed_mask
                scores[name] = compare(logits,candidate)
                assert torch.equal(caches[name]['next_text_positions'],initial_physical_position+step+1)
            result['steps'].append(scores)
        (OUTPUT/f'row_{shard}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    jobs, logs = [], []
    try:
        for i in range(8):
            log=(OUTPUT/f'worker{i}.log').open('w');logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_cached_decode_regression',str(i)],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT))
        codes=[p.wait() for p in jobs]
        assert not any(codes),codes
    finally:
        for log in logs:log.close()
    rows=[json.loads((OUTPUT/f'row_{i}.json').read_text()) for i in range(8)]
    summary={name:dict(comparisons=32,
        mean_kl=sum(s[name]['kl'] for r in rows for s in r['steps'])/32,
        max_kl=max(s[name]['kl'] for r in rows for s in r['steps']),
        matching_argmax=sum(s[name]['same_argmax'] for r in rows for s in r['steps']))
        for name in ['legacy','fixed','hf_static']}
    summary['visibility']=[{k:r[k] for k in ['index','visual_tokens','legacy_visible_visual_tokens','fixed_visible_visual_tokens']}
                           for r in rows]
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
