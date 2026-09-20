"""Semantics-preserving permutation tests on the current 184 matching cases.

No training, no checkpoint/benchmark mutation. All original cases are included,
not only errors. Report image-choice and textual-choice tasks separately.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/diagnostics/muir_random1000_seed42_matched_20260914'
OUT = ROOT / 'artifacts/diagnostics/muir_random_matching_permutations_20260914'
METHODS = ('base', 'embedding_adapter', 'embedding_adapter_mixed')
LAYOUTS = ('original', 'rotate_choices_fixed_media', 'rotate_media_fixed_choices')


def image_position(prediction, choices):
    if prediction is None or len(prediction) != 1 or not 0 <= ord(prediction)-65 < len(choices):
        return None
    match = re.fullmatch(r'<\|image_(\d+)\|>', choices[ord(prediction)-65].strip())
    return int(match[1]) if match else None


def worker(shard):
    import src
    src.__path__.insert(0, str(ROOT.parent / 'vision-kv-inject-attention-sink/src'))
    import torch
    from src import model as ref
    from src.data import QwenBenchmarkDataset
    from src.multimodal_baseline_suite import MODEL, ADAPTER_CHECKPOINTS
    from src.muir_binding_diagnostic import permute_row
    from src.audit_mmiu_random_results import extract_answer
    from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item

    torch.set_num_threads(4); torch.manual_seed(42)
    model, processor = load_baseline_model('base', MODEL, torch.bfloat16, 'cuda:0', 1., 'sdpa')
    model.eval().requires_grad_(False)
    adapters = {}
    metadata = {}
    for method in METHODS[1:]:
        adapter, meta = ref.load_qwen_embedding_adapter_checkpoint(
            ADAPTER_CHECKPOINTS[method], model.model.language_model, torch.device('cuda'), torch.bfloat16)
        assert not meta['missing'] and not meta['unexpected']
        assert adapter.adapter_start_layer == 0 and adapter.active_adapter_layers == 0
        assert adapter.mode == 'embedding_adapter' and not adapter.native_ffn_carriers
        adapters[method] = adapter.eval().requires_grad_(False)
        metadata[method] = {'checkpoint': str(ADAPTER_CHECKPOINTS[method]), 'load_meta': meta}
    model.model.language_model.register_forward_pre_hook(
        lambda m,a,k: (a,dict(k,deepstack_visual_embeds=None)), with_kwargs=True)
    def reject(*a, **kw): raise AssertionError('DeepStack executed')
    model.model.language_model._deepstack_process = reject
    original_vision = model.model.visual.forward
    vision_cache = {}
    def cached_vision(*a, **k):
        if not vision_cache:
            value = original_vision(*a, **k)
            vision_cache['value'] = (type(value), dict(value))
        cls, fields = vision_cache['value']
        return cls(**fields)
    model.model.visual.forward = cached_vision
    ds = QwenBenchmarkDataset(str(SOURCE/'muirbench_random1000.jsonl'), processor, 'muirbench',
        data_root=str(ROOT/'data/benchmarks/muirbench'), max_samples=1000, prompt_layout='media_first_v1')
    indices = [i for i,r in enumerate(ds.rows) if r['task'] == 'Image-Text Matching']
    assert len(indices) == 184
    old = {(r['method'], r['index']):r for method in METHODS
           for p in SOURCE.glob(method+'_shard*.jsonl') for line in p.open() if (r:=json.loads(line))}

    with torch.inference_mode(), (OUT/f'rows_{shard}.jsonl').open('w',buffering=1) as output:
        for index in indices[shard::8]:
            original = ds.rows[index]
            # These cases contain no implicit positional references requiring
            # remapping beyond their explicit image placeholders.
            assert not re.search(r'(?:first|second|third|fourth|left|right)\s+(?:image|picture)|'
                                 r'(?:image|picture)\s*[1-9]', original['question'], re.I)
            image_choices = any(re.fullmatch(r'<\|image_\d+\|>', c.strip()) for c in original['choices'])
            for layout in LAYOUTS:
                row = original if layout == 'original' else permute_row(original, layout)
                ds.rows[index] = row
                vision_cache.clear()
                item = ds[index]
                inputs0 = _qwen_inputs_from_item(item, torch.device('cuda'))
                initial, positions = ref.build_qwen_initial_context(model, inputs0)
                for method in METHODS:
                    inputs = dict(inputs0); h = initial; pos = positions; generated = []
                    eos = model.generation_config.eos_token_id
                    eos = eos if isinstance(eos, list) else [eos]
                    for step in range(8):
                        model.model.rope_deltas = None
                        if method == 'base':
                            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits
                        else:
                            logits = ref.qwen_embedding_adapter_logits(model, adapters[method], inputs,
                                initial_hidden=h, position_ids=pos, logits_to_keep=1)[0]
                        token = int(logits[0,-1].argmax()); generated.append(token)
                        text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
                        if token in eos or text in [chr(65+j) for j in range(len(row['choices']))]: break
                        new = torch.tensor([[token]],device='cuda',dtype=inputs['input_ids'].dtype)
                        inputs['input_ids'] = torch.cat([inputs['input_ids'],new],1)
                        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                        inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(new)],1)
                        h = torch.cat([h,model.model.get_input_embeddings()(new)],1)
                        pos = torch.cat([pos,pos[:,:,-1:]+1],2)
                    prediction = extract_answer(text, row['choices'])
                    prior = old[method,index]
                    if layout == 'original':
                        assert text == prior['text'].strip(), (method,index,'Original run did not reproduce',text,prior['text'])
                    expected = prior['prediction']
                    if layout == 'rotate_choices_fixed_media' and expected is not None:
                        expected = chr(65+(ord(expected)-66)%len(row['choices']))
                    before_position = image_position(prior['prediction'], original['choices'])
                    after_position = image_position(prediction, row['choices'])
                    output.write(json.dumps(dict(index=index,source_index=original['index'],method=method,
                        layout=layout,image_count=len(row['images']),image_choices=image_choices,
                        text=text,prediction=prediction,gold=row['answer'],score=int(prediction==row['answer']),
                        original_prediction=prior['prediction'],original_score=prior['score'],
                        same_content_choice=prediction is not None and prediction==expected,
                        expected_prediction=expected,original_image_position=before_position,
                        predicted_image_position=after_position,
                        same_display_position=(before_position==after_position) if before_position is not None and after_position is not None else None,
                        generated_tokens=len(generated),deepstack=False,choices=row['choices'],images=row['images']))+'\n')
            ds.rows[index] = original
            print('DONE',shard,index,flush=True)


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    logs=[];jobs=[]
    for shard in range(8):
        log=(OUT/f'worker{shard}.log').open('w');logs.append(log)
        jobs.append(subprocess.Popen([sys.executable,'-m','src.muir_random_matching_permutation_audit',str(shard)],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    for f in logs:f.close()
    rows=[json.loads(l) for p in OUT.glob('rows_*.jsonl') for l in p.open()]
    results=[]
    for subset in ('all','image_choices','text_choices'):
        for method in METHODS:
            for layout in LAYOUTS:
                group=[r for r in rows if r['method']==method and r['layout']==layout and
                       (subset=='all' or r['image_choices']==(subset=='image_choices'))]
                positions=[r for r in group if r['same_display_position'] is not None]
                if not group:continue
                results.append(dict(subset=subset,method=method,layout=layout,n=len(group),
                    accuracy=100*sum(r['score'] for r in group)/len(group),
                    same_content_choice=100*sum(r['same_content_choice'] for r in group)/len(group),
                    both_choose_images=len(positions),
                    same_display_position=100*sum(r['same_display_position'] for r in positions)/len(positions) if positions else None))
    payload=dict(expected=1656,completed=len(rows),exit_codes=codes,results=results)
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2),flush=True)
    assert not any(codes) and len(rows)==len({(r['index'],r['method'],r['layout']) for r in rows})==1656


if __name__=='__main__': run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
