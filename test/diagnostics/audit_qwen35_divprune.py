"""Independent physical-deletion and actual mixer-input audit for Qwen3.5 DivPrune."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'artifacts/dependencies/qwen35_python'))
import torch
from src.qwen35_experiment import load_model, prepare_inputs, initial_context, dump
from src.qwen35_pruning import VisualPruningController, visual_budget
from src.benchmarks import get_benchmark_spec, build_benchmark_prompt
from scripts.qwen35_worker import compare_caches, read_rows


def predictions(run, method, name):
    return {p['index']:p for file in (run/'eval'/method).glob(f'{name}.shard*.jsonl') for p in read_rows(file)}


def independent_selection(features, count):
    # Straightforward full selected-row reduction, not the incremental port.
    features = torch.nn.functional.normalize(features.float(), dim=-1)
    distance = 1 - features @ features.T
    selected = []
    for step in range(count):
        scores = (distance.topk(2, dim=0, largest=False).values[1] if not selected else
                  distance[selected].min(dim=0).values)
        if selected:
            scores[selected] = -float('inf')
        selected.append(int(scores.argmax()))
    return torch.tensor(selected, device=features.device)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    args = ap.parse_args()
    run = args.run_dir
    cfg = json.loads((run/'config.json').read_text())
    processor, model, adapter, old = load_model(cfg, torch.device('cuda:0'))
    old.close()
    del old, adapter
    control = VisualPruningController(model)
    report = {'checks':[], 'passed':False}
    for name in ('mmstar','gqa','mmb','mmb-cn'):
        info = cfg['evaluation'][name]
        rows = read_rows(info['path'])
        native = predictions(Path(cfg['reference_run']), 'native', name)
        prior20 = predictions(run, 'divprune_20', name)
        wins = [i for i in sorted(prior20) if prior20[i]['score'] > native[i]['score']]
        indices = list(dict.fromkeys([0]+wins[:1]))
        for index in indices:
            row = rows[index]
            spec = get_benchmark_spec(name)
            inputs, _ = prepare_inputs(processor,row,info['image_root'],torch.device('cuda:0'),
                                      question=build_benchmark_prompt(row,spec))
            mask = inputs['mm_token_type_ids'].eq(1)
            assert torch.equal(mask, inputs['input_ids'].eq(model.config.image_token_id))
            assert int(mask.sum()) == int(inputs['image_grid_thw'].prod(-1).sum())//4
            context = initial_context(model, inputs)
            vision = mask[0].nonzero().flatten()
            text = (~mask[0]).nonzero().flatten()
            for ratio in (.05,.2):
                original = predictions(run, f'divprune_{round(ratio*100)}', name)[index]
                actual = {}
                handles = []
                def record(key, module, args, kwargs):
                    h = args[0] if args else kwargs['hidden_states']
                    actual.setdefault(key, []).append(h.shape[1])
                from functools import partial
                for li,layer in enumerate(model.model.language_model.layers):
                    handles.append(layer.register_forward_pre_hook(partial(record, f'layer{li}'),with_kwargs=True))
                    mixer = layer.self_attn if hasattr(layer,'self_attn') else layer.linear_attn
                    handles.append(mixer.register_forward_pre_hook(partial(record, f'mixer{li}'),with_kwargs=True))
                with control.activate('divprune',ratio,mask):
                    hooked = model.model.language_model(**dict(context,use_cache=True))
                for h in handles:
                    h.remove()
                audit = dict(control.audit)
                selected = torch.tensor(audit['selected_visual_indices'],device=vision.device)
                assert audit['selected_visual_indices'] == original['token_audit']['selected_visual_indices']
                independently_selected = vision[independent_selection(context['inputs_embeds'][0,vision],visual_budget(vision.numel(),ratio))].sort().values
                assert torch.equal(selected,independently_selected)
                keep = torch.cat((text,selected)).sort().values
                assert len(actual)==64 and all(lengths==[keep.numel()] for lengths in actual.values()), actual
                # No controller is active: pass ONLY the physically retained sequence to native decoder.
                cropped = dict(inputs_embeds=context['inputs_embeds'][:,keep],
                    position_ids=context['position_ids'][...,keep],
                    attention_mask=torch.ones((1,keep.numel()),device=vision.device,dtype=torch.long),use_cache=True)
                physical = model.model.language_model(**cropped)
                torch.testing.assert_close(hooked.last_hidden_state,physical.last_hidden_state,rtol=0,atol=0)
                cache_count = compare_caches(hooked.past_key_values,physical.past_key_values)
                cache_lengths={str(i):int(hooked.past_key_values.get_seq_length(i)) for i in (3,7,11,15,19,23,27,31)}
                assert all(v==keep.numel() for v in cache_lengths.values())
                # Fix selection, destroy every discarded row, and verify zero downstream effect.
                dropped = torch.ones(mask.shape[1],device=vision.device,dtype=torch.bool)
                dropped[keep] = False
                corrupted = context['inputs_embeds'].clone()
                corrupted[:,dropped] = 1234.
                with control.activate('divprune',ratio,mask,fixed_visual=selected):
                    changed = model.model.language_model(**dict(context,inputs_embeds=corrupted,use_cache=True))
                torch.testing.assert_close(hooked.last_hidden_state,changed.last_hidden_state,rtol=0,atol=0)
                compare_caches(hooked.past_key_values,changed.past_key_values)
                # Reproduce the actual generated answer (not merely a logged token count).
                with control.activate('divprune',ratio,mask):
                    output = model.generate(**inputs,do_sample=False,max_new_tokens=spec.max_new_tokens,
                                             use_cache=True,pad_token_id=processor.tokenizer.pad_token_id)
                answer = processor.tokenizer.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
                assert answer == original['prediction_text'], (name,index,ratio,answer,original['prediction_text'])
                check=dict(benchmark=name,index=index,retention=ratio,original_visual=int(vision.numel()),
                    retained_visual=int(selected.numel()),text_tokens=int(text.numel()),actual_input_lengths=actual,
                    full_attention_cache_lengths=cache_lengths,exact_cache_tensors=cache_count,
                    physical_deletion_exact=True,dropped_token_corruption_exact=True,selector_matches=True,
                    saved_answer_reproduced=True,prediction_text=answer,score=original['score'],native_score=native[index]['score'])
                report['checks'].append(check)
                dump(run/'divprune_runtime_audit.json',report)
                print(json.dumps({k:v for k,v in check.items() if k not in ('actual_input_lengths','full_attention_cache_lengths')}),flush=True)
                del hooked,physical,changed
    report['passed']=True
    dump(run/'divprune_runtime_audit.json',report)
    print('ALL_PASSED',flush=True)


if __name__=='__main__':
    main()
