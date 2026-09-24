"""Native Qwen forward with visual hidden reset to initial E before every layer."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch


def dump(path, value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


class InitialVisualReset:
    def __init__(self,model):
        self.enabled=False
        self.layers=None
        self.handles=[layer.register_forward_pre_hook(self.hook(i),with_kwargs=True)
                      for i,layer in enumerate(model.model.language_model.layers)]

    def begin(self,inputs,*,enabled,layers=None,validate=False):
        self.enabled=enabled;self.layers=layers;self.validate=validate
        self.positions=inputs['mm_token_type_ids'][0].ne(0).nonzero().flatten()
        self.prefill_length=inputs['input_ids'].shape[1]
        self.initial=None;self.counts=[0]*len(self.handles);self.changed_layers=[]
        assert self.positions.numel()>0 and inputs['input_ids'].shape[0]==1

    def hook(self,index):
        def reset(module,args,kwargs):
            if not self.enabled:return
            h=kwargs.get('hidden_states',args[0] if args else None)
            if h.shape[1]!=self.prefill_length:
                assert h.shape[1]==1,'Unexpected forward sequence length'
                return  # Decode uses visual K/V already built from E during prefill.
            if index==0:self.initial=h[:,self.positions].detach().clone()
            assert self.initial is not None
            if self.layers is not None and index not in self.layers:return
            restored=h.clone()
            restored[:,self.positions]=self.initial.to(restored)
            if self.validate:
                text=torch.ones(h.shape[1],device=h.device,dtype=torch.bool);text[self.positions]=False
                assert torch.equal(restored[:,text],h[:,text])
                assert torch.equal(restored[:,self.positions],self.initial)
                if not torch.equal(h[:,self.positions],self.initial):self.changed_layers.append(index)
            self.counts[index]+=1
            return (args,dict(kwargs,hidden_states=restored)) if 'hidden_states' in kwargs else ((restored,)+args[1:],kwargs)
        return reset


def inputs_from_item(item,device):
    inputs={k:item[k].unsqueeze(0).to(device) for k in ('input_ids','attention_mask','mm_token_type_ids')}
    inputs.update({k:item[k].to(device) for k in ('pixel_values','image_grid_thw') if torch.is_tensor(item.get(k))})
    return inputs


def audit(model,reset,inputs):
    reset.begin(inputs,enabled=False)
    model.model.rope_deltas=None
    native=model(**inputs,use_cache=False,logits_to_keep=1).logits
    reset.begin(inputs,enabled=True,layers={0},validate=True)
    model.model.rope_deltas=None
    layer0=model(**inputs,use_cache=False,logits_to_keep=1).logits
    assert torch.equal(native,layer0),'Resetting only layer0 must be a no-op'
    reset.begin(inputs,enabled=True,validate=True)
    model.model.rope_deltas=None
    output=model(**inputs,use_cache=True,logits_to_keep=1)
    assert reset.counts==[1]*len(reset.handles),reset.counts
    assert reset.changed_layers,'Visual evolution was not intercepted'
    token=output.logits[:,-1].argmax(-1).view(1,1)
    # Run the genuine HF generation preparation on the following decode token.
    assert output.past_key_values.get_seq_length()==inputs['input_ids'].shape[1]
    return dict(layer0_noop_bitwise_equal=True,all_layers_reset=True,
        text_rows_bitwise_preserved=True,changed_layers=reset.changed_layers,
        visual_tokens=len(reset.positions),native_first_token=int(native[0,-1].argmax()),
        reset_first_token=int(token),deepstack=model._benchmark_deepstack)


def worker(args):
    from src.model import load_frozen_qwen3vl
    from src.data import QwenBenchmarkDataset
    from src.benchmarks import get_benchmark_spec,score_prediction
    run=Path(args.run_dir);config=json.loads((run/'config.json').read_text())
    torch.set_num_threads(4);torch.manual_seed(44)
    processor,model=load_frozen_qwen3vl(config['model_path'],torch.bfloat16,torch.device('cuda:0'),'flash_attention_2')
    reset=InitialVisualReset(model)
    assert model.model.visual.deepstack_visual_indexes==[]
    assert model.model.language_model.config._attn_implementation=='flash_attention_2'
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    out=run/'rows'/f'shard{args.shard}.jsonl'
    done=set()
    if out.exists():
        for line in out.read_text().splitlines():
            r=json.loads(line);done.add((r['benchmark'],r['sample'],r['method']))
    start=time.time()
    with torch.inference_mode(),out.open('a',buffering=1) as stream:
        for name,info in config['evaluation'].items():
            dataset=QwenBenchmarkDataset(info['path'],processor,name,data_root=info['image_root'])
            assert len(dataset)==info['samples']
            audited=False
            for index in range(args.shard,len(dataset),config['shards']):
                if all((name,index,m) in done for m in config['methods']):continue
                item=dataset[index];row=item['row'];inputs=inputs_from_item(item,torch.device('cuda:0'))
                if not audited:
                    result=audit(model,reset,inputs)
                    dump(run/'audits'/f'{name}_shard{args.shard}.json',result);audited=True
                digest=hashlib.sha256()
                for key,value in sorted(inputs.items()):
                    v=value.cpu().contiguous();digest.update(str((key,list(v.shape),str(v.dtype))).encode());digest.update(v.view(torch.uint8).numpy().tobytes())
                for method in config['methods']:
                    if (name,index,method) in done:continue
                    reset.begin(inputs,enabled=method=='initial_embedding')
                    model.model.rope_deltas=None
                    generated=model.generate(**inputs,do_sample=False,max_new_tokens=info['max_new_tokens'],use_cache=True)
                    tokens=generated[0,inputs['input_ids'].shape[1]:].tolist()
                    if method=='initial_embedding':assert reset.counts==[1]*36,reset.counts
                    text=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip()
                    scored=score_prediction(metric=get_benchmark_spec(name).metric,prediction_text=text,
                        answer=row.get('answer'),answers=row.get('answers'),choices=row.get('choices'),question=row.get('question'))
                    record=dict(benchmark=name,sample=index,source_index=row.get('index'),method=method,
                        prediction_text=text,generated_token_ids=tokens,stopped_by_eos=bool(tokens and tokens[-1] in eos),
                        max_new_tokens=info['max_new_tokens'],input_sha256=digest.hexdigest(),**scored)
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                if index%80<config['shards']:print(name,index,'elapsed',round(time.time()-start),flush=True)
    dump(run/'rows'/f'shard{args.shard}.done.json',dict(complete=True,elapsed_seconds=time.time()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--shard',type=int,required=True)
    worker(p.parse_args())
