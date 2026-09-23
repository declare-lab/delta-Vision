"""DivPrune accuracy: exact historical multimodal inputs and seed44 single images."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def input_digest(item):
    digest = hashlib.sha256()
    for name in ('input_ids','attention_mask','mm_token_type_ids','pixel_values','image_grid_thw','pixel_values_videos','video_grid_thw'):
        if torch.is_tensor(item.get(name)):
            v = item[name].contiguous().cpu()
            digest.update(str((name,tuple(v.shape),str(v.dtype))).encode())
            digest.update(v.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class LayerAudit:
    def __init__(self, model):
        self.lengths = {}
        self.handles = [layer.register_forward_pre_hook(self.hook(i), with_kwargs=True)
                        for i,layer in enumerate(model.model.language_model.layers)]
    def hook(self, index):
        def record(module, args, kwargs):
            if index not in self.lengths:
                h = kwargs.get('hidden_states', args[0] if args else None)
                self.lengths[index] = int(h.shape[1])
        return record
    def verify(self, visual, text, retained):
        assert len(self.lengths) == len(self.handles), self.lengths
        assert all(n == text+retained for n in self.lengths.values()), (visual,text,retained,self.lengths)
        return dict(original_visual=visual, retained_visual=retained, text=text,
                    visual_tokens_per_layer=[self.lengths[i]-text for i in range(len(self.handles))],
                    all_layer_visual_retention=retained/visual)


def worker(args):
    from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
    from baselines import llava_hf_baselines as llava
    from baselines.multimodal_pruning_utils import visual_budget
    from src.data import QwenBenchmarkDataset, LlavaBenchmarkDataset
    from src.benchmarks import get_benchmark_spec, score_prediction
    from src.qwen_deepstack import disable_qwen_deepstack
    run = Path(args.run_dir)
    config = json.loads((run/'config.json').read_text())
    model_info = config['models'][args.model]
    multi = args.model == 'qwen3-vl-4b'
    datasets = config['multimodal'] if multi else config['single_image']
    torch.set_num_threads(4)
    torch.manual_seed(44)
    torch.backends.cuda.matmul.allow_tf32 = False
    if model_info['kind'] == 'qwen':
        model, processor = load_baseline_model('divprune', model_info['path'], torch.bfloat16,'cuda:0',.05,'flash_attention_2')
        disable_qwen_deepstack(model)
    else:
        processor, model = llava.load_llava_baseline_model(model_info['path'], dtype=torch.bfloat16,
                                                         device='cuda:0', attn_implementation='flash_attention_2')
    model.eval().requires_grad_(False)
    assert model.model.language_model.config._attn_implementation == 'flash_attention_2'
    audit = LayerAudit(model)
    folder = run/args.phase/args.model
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/f'shard{args.shard}.jsonl'
    # Resume only complete saved request/retention pairs.
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line); done.add((row['benchmark'],row['sample'],row['retention']))
    cache = json.loads((run/'multimodal_input_cache.json').read_text()) if multi else None
    original_reduce = llava.reduce_visual_memory
    counts = {}
    generated_record = {}
    if model_info['kind'] == 'llava':
        original_generate = model.generate
        def capture_generate(*positional, **kwargs):
            result = original_generate(*positional, **kwargs)
            # HF generation from inputs_embeds returns only the generated IDs.
            assert 'inputs_embeds' in kwargs and 'input_ids' not in kwargs
            generated_record['tokens'] = result[0].tolist()
            eos = kwargs.get('eos_token_id', model.generation_config.eos_token_id)
            generated_record['eos'] = eos if isinstance(eos, list) else [eos]
            return result
        model.generate = capture_generate
    def reduce(memory, *, method, retention):
        result = original_reduce(memory,method=method,retention=retention)
        counts.update(visual=int(memory.shape[1]),kept=int(result.shape[1]))
        return result
    llava.reduce_visual_memory = reduce
    started = time.time()
    with torch.inference_mode(), path.open('a',buffering=1) as output:
        for name,info in datasets.items():
            metric = 'multi_choice' if multi else get_benchmark_spec(name).metric
            rows = [json.loads(s) for s in Path(info['path']).read_text().splitlines() if s]
            cls = QwenBenchmarkDataset if model_info['kind']=='qwen' else LlavaBenchmarkDataset
            dataset_kwargs = ({'prompt_template': config.get('llava_prompt_template', 'auto')}
                              if model_info['kind']=='llava' else {})
            dataset = None if multi else cls(info['path'],processor,name,data_root=info['image_root'],**dataset_kwargs)
            indices = range(min(2,len(rows))) if args.phase=='smoke' else range(args.shard,len(rows),args.shards)
            for index in indices:
                if all((name,index,r) in done for r in (.05,.2)):continue
                row = rows[index]
                if multi:
                    entry = cache[name][index]
                    item = torch.load(entry['path'],map_location='cpu',weights_only=False,mmap=True)['item']
                    assert item['row']['question']==row['question'] and item['answer']==row.get('answer')
                    assert input_digest(item)==entry['input_sha256'], (name,index,'historical input mismatch')
                    digest = entry['input_sha256']
                else:
                    item = dataset[index]
                    digest = input_digest(item)
                for retention in (.05,.2):
                    if (name,index,retention) in done:continue
                    audit.lengths.clear()
                    begin = time.time()
                    if model_info['kind']=='qwen':
                        inputs = _qwen_inputs_from_item(item,torch.device('cuda:0'))
                        visual = inputs['mm_token_type_ids'][0].ne(0).nonzero().flatten()
                        assert len(visual)>0
                        configure_baseline(model,'divprune',retention,int(visual[0]),len(visual))
                        if hasattr(model.model,'rope_deltas'):model.model.rope_deltas=None
                        if multi:
                            # Same fresh-prefix and standalone-option stopping as the historical multimodal suite.
                            current=dict(inputs); tokens=[]
                            eos=model.generation_config.eos_token_id
                            eos=eos if isinstance(eos,list) else [eos]
                            stop='length'
                            for _ in range(info['max_new_tokens']):
                                model.model.rope_deltas=None
                                logits=model(**current,use_cache=False,logits_to_keep=1).logits
                                token=int(logits[0,-1].argmax());tokens.append(token)
                                text=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
                                if token in eos:stop='eos';break
                                if text in [chr(65+i) for i in range(len(row.get('choices') or []))]:stop='standalone_option';break
                                new=torch.tensor([[token]],device='cuda:0')
                                current['input_ids']=torch.cat((current['input_ids'],new),1)
                                current['attention_mask']=torch.ones_like(current['input_ids'])
                                current['mm_token_type_ids']=torch.cat((current['mm_token_type_ids'],torch.zeros_like(new)),1)
                        else:
                            generated=model.generate(**inputs,do_sample=False,max_new_tokens=info['max_new_tokens'],use_cache=True)
                            tokens=generated[0,inputs['input_ids'].shape[1]:].tolist()
                            text=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
                            eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                            stop='eos' if tokens and tokens[-1] in eos else 'length'
                        token_audit=audit.verify(len(visual),inputs['input_ids'].shape[1]-len(visual),visual_budget(len(visual),retention))
                    else:
                        ids=item['input_ids'].unsqueeze(0).cuda()
                        image_id=int(model.config.image_token_index)
                        sizes=item.get('image_sizes')
                        if torch.is_tensor(sizes):sizes=sizes.unsqueeze(0).cuda()
                        text=llava.generate_llava_baseline(model,processor,input_ids=ids,
                            attention_mask=item['attention_mask'].unsqueeze(0).cuda(),
                            pixel_values=item['pixel_values'].unsqueeze(0).cuda(),image_token_id=image_id,
                            method='divprune',retention=retention,max_new_tokens=info['max_new_tokens'],image_sizes=sizes)
                        assert counts['kept']==visual_budget(counts['visual'],retention)
                        token_audit=audit.verify(counts['visual'],int((ids!=image_id).sum()),counts['kept'])
                        tokens=generated_record['tokens']
                        assert text == processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
                        stop='eos' if tokens and tokens[-1] in generated_record['eos'] else 'length'
                    scored=score_prediction(metric=metric,prediction_text=text,answer=row.get('answer'),
                        answers=row.get('answers'),choices=row.get('choices'),question=row.get('question'))
                    record=dict(model=args.model,benchmark=name,sample=index,source_index=row.get('index'),
                        retention=retention,text=text,**scored,token_audit=token_audit,input_sha256=digest,
                        seconds=time.time()-begin,max_new_tokens=info['max_new_tokens'],generated_token_ids=tokens,stop=stop,
                        stopped_by_eos=stop=='eos',
                        hit_generation_limit=stop=='length',
                        fa2=True,deepstack=False)
                    output.write(json.dumps(record,ensure_ascii=False)+'\n')
                if index%20<args.shards:
                    print(args.model,name,args.shard,index,round(time.time()-started,1),flush=True)
    dump(folder/f'shard{args.shard}.done.json',dict(completed=True,seconds=time.time()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--model',required=True)
    p.add_argument('--phase',choices=['smoke','full'],required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2)
    worker(p.parse_args())
