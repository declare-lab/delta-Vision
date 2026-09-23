"""Audit physical pruning and cached generation against saved DivPrune answers."""
import argparse
import json
from pathlib import Path
import torch
from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
from baselines.multimodal_pruning_utils import visual_budget
from src.data import QwenBenchmarkDataset
from src.qwen_deepstack import disable_qwen_deepstack


def independent_select(features, count):
    f = torch.nn.functional.normalize(features.float(), dim=-1)
    distance = 1 - f @ f.T
    chosen = []
    for _ in range(count):
        scores = distance.topk(2, dim=0, largest=False).values[1] if not chosen else distance[chosen].min(0).values
        scores = scores.clone()
        if chosen:
            scores[chosen] = -float('inf')
        chosen.append(int(scores.argmax()))
    return torch.tensor(chosen, device=features.device)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--model', required=True)
    a = p.parse_args()
    config = json.loads((a.run_dir/'config.json').read_text())
    info = config['models'][a.model]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, processor = load_baseline_model('divprune', info['path'], torch.bfloat16, 'cuda:0', .05, 'flash_attention_2')
    disable_qwen_deepstack(model)
    lm = model.model.language_model
    saved = {}
    for path in (a.run_dir/'full'/a.model).glob('*.jsonl'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            saved[(row['benchmark'], row['sample'], row['retention'])] = row
    report = []
    for benchmark, index in [('mmb-cn', 28), ('gqa', 26)]:
        spec = config['single_image'][benchmark]
        dataset = QwenBenchmarkDataset(spec['path'], processor, benchmark, data_root=spec['image_root'])
        item = dataset[index]
        inputs = _qwen_inputs_from_item(item, torch.device('cuda:0'))
        visual = inputs['mm_token_type_ids'][0].ne(0).nonzero().flatten()
        for ratio in [.05, .2]:
            configure_baseline(model, 'divprune', ratio, int(visual[0]), len(visual))
            model.model.rope_deltas = None
            recorded = {}
            def capture_context(module, args, kwargs):
                recorded['context'] = {k:v for k,v in kwargs.items() if k in ['inputs_embeds','position_ids']}
            def capture_first(module, args, kwargs):
                recorded['layer0'] = (args[0] if args else kwargs['hidden_states']).clone()
            h0 = lm.register_forward_pre_hook(capture_context, with_kwargs=True)
            h1 = lm.layers[0].register_forward_pre_hook(capture_first, with_kwargs=True)
            original = model(**inputs, use_cache=True, logits_to_keep=1)
            h0.remove(); h1.remove()
            context = recorded['context']
            kept_visual = visual[independent_select(context['inputs_embeds'][0, visual], visual_budget(len(visual), ratio))]
            keep_mask = torch.ones(inputs['input_ids'].shape[1], device='cuda:0', dtype=torch.bool)
            keep_mask[visual] = False; keep_mask[kept_visual] = True
            keep = keep_mask.nonzero().flatten()
            physical_embeddings = context['inputs_embeds'][:, keep]
            torch.testing.assert_close(recorded['layer0'], physical_embeddings, rtol=0, atol=0)
            old = lm.config.divprune_config
            lm.config.divprune_config = None
            physical = lm(inputs_embeds=physical_embeddings, position_ids=context['position_ids'][..., keep],
                          attention_mask=None, use_cache=True)
            lm.config.divprune_config = old
            logits = model.lm_head(physical.last_hidden_state[:, -1:])
            torch.testing.assert_close(logits, original.logits, rtol=0, atol=0)
            cache_lengths = [original.past_key_values.get_seq_length(i) for i in range(len(lm.layers))]
            assert cache_lengths == [len(keep)]*len(lm.layers)
            lengths = {i:[] for i in range(len(lm.layers))}
            def hook(i):
                def record(module, args, kwargs):
                    lengths[i].append((args[0] if args else kwargs['hidden_states']).shape[1])
                return record
            handles = [layer.register_forward_pre_hook(hook(i), with_kwargs=True) for i,layer in enumerate(lm.layers)]
            model.model.rope_deltas = None
            output = model.generate(**inputs, do_sample=False, max_new_tokens=spec['max_new_tokens'],
                                    use_cache=True, return_dict_in_generate=True)
            for handle in handles:handle.remove()
            tokens = output.sequences[0, inputs['input_ids'].shape[1]:].tolist()
            previous = saved[(benchmark,index,ratio)]
            assert tokens == previous['generated_token_ids'], (benchmark,index,ratio,tokens,previous['generated_token_ids'])
            assert all(v == [len(keep)]+[1]*(len(tokens)-1) for v in lengths.values()), lengths
            generated_cache = [output.past_key_values.get_seq_length(i) for i in range(len(lm.layers))]
            assert generated_cache == [len(keep)+len(tokens)-1]*len(lm.layers)
            # Compare cached generation with fresh-prefix inference for three steps.
            fresh = dict(inputs)
            fresh_tokens = []
            for t in tokens[:3]:
                model.model.rope_deltas = None
                pred = int(model(**fresh,use_cache=False,logits_to_keep=1).logits[0,-1].argmax())
                fresh_tokens.append(pred)
                assert pred == t, (benchmark,index,ratio,fresh_tokens,tokens)
                new = torch.tensor([[t]],device='cuda:0')
                fresh['input_ids'] = torch.cat([fresh['input_ids'],new],1)
                fresh['attention_mask'] = torch.ones_like(fresh['input_ids'])
                fresh['mm_token_type_ids'] = torch.cat([fresh['mm_token_type_ids'],torch.zeros_like(new)],1)
            result = dict(benchmark=benchmark,sample=index,retention=ratio,original_visual=len(visual),
                kept_visual=len(kept_visual),physical_pruning_exact=True,physical_logits_exact=True,
                prefill_cache_lengths=cache_lengths,decode_cache_lengths=generated_cache,
                layer_input_lengths=lengths,saved_token_ids_reproduced=True,cached_fresh_prefix_tokens_equal=fresh_tokens,
                text=processor.tokenizer.decode(tokens,skip_special_tokens=True))
            report.append(result)
            out = a.run_dir/'audit'/f'runtime_{a.model}.json'
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
            print(benchmark,index,ratio,'PASSED',repr(result['text']),flush=True)
    print('ALL_PASSED',flush=True)


if __name__ == '__main__':main()
