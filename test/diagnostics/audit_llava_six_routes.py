"""Trace the active LLaVA routes and compare selectors to vendored Git HEAD.

CPU-only audit. Synthetic tensors test selector fidelity, NOT task accuracy.
Production model/evaluation code is never modified by this script.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
import torch

from baselines import llava_hf_baselines as llava
from baselines.llava_dart_corrected import select_dart

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/reports/llava_six_author_audit_20260922'

def source(method,path):
    repo=ROOT/'baselines'/method
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    raw=subprocess.check_output(['git','-C',str(repo),'show','HEAD:'+path],text=True)
    return raw,dict(commit=commit,path=path,sha256=hashlib.sha256(raw.encode()).hexdigest())

def extract(raw,name,env):
    node=next(n for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.FunctionDef) and n.name==name)
    node.returns=None
    for a in node.args.args:a.annotation=None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),name,'exec'),env)
    return env[name]

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(44)
    refs={}
    paths={'dart':'llava/model/language_model/modeling_llama_self.py',
        'divprune':'LLaVA/llava/model/llava_arch.py',
        'fastv':'src/FastV/llava-hf/transformers/src/transformers/models/llama/modeling_llama.py',
        'sparsevlm':'llava/model/language_model/modelling_sparse_llama.py',
        'visionzip':'visionzip/clip_encoder.py','zoo':'LLaVA/llava/model/llava_arch.py'}
    originals={}
    for m,p in paths.items():originals[m],refs[m]=source(m,p)

    # Exercise the actual entry, with only model compute replaced by spies.
    # This distinguishes defined-but-unused layer-pruning helpers from callers.
    route=[];events=[]
    class Model:
        def __init__(self):self.model=SimpleNamespace(language_model=SimpleNamespace(embed_tokens=torch.nn.Embedding(128,32)))
        def generate(self,**kw):
            events.append(dict(call='native_generate',compressed_input_length=kw['inputs_embeds'].shape[1]))
            return torch.tensor([[7,2]])
    class Decoder:
        def __init__(self,model):pass
        def generate(self,emb,start,length,retention,cap,eos):
            events.append(dict(call='DartDecoder',input_length=emb.shape[1],visual_count=length))
            return [7,2],{}
    tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=2,decode=lambda ids,**kw:'answer')
    processor=SimpleNamespace(tokenizer=tokenizer);memory=torch.randn(1,576,32)
    ids=torch.tensor([[1]+[99]*576+[3,4]])
    original_reduce=llava.reduce_visual_memory
    def reduce(mem,**kw):
        events.append(dict(call='reduce_visual_memory',**kw))
        result=original_reduce(mem,**kw)
        events.append(dict(call='reduced',visual_count=result.shape[1],
                           only_original_rows=bool(torch.isin(result[0,:,0],mem[0,:,0]).all())))
        return result
    with patch.object(llava,'llava_projected_image_features',return_value=memory), \
         patch.object(llava,'_get_language_model',side_effect=lambda m:m.model.language_model), \
         patch.object(llava,'reduce_visual_memory',side_effect=reduce), \
         patch('baselines.llava_dart_corrected.DartDecoder',Decoder):
        for m in ['fastv','dart','sparsevlm','visionzip','divprune','zoo']:
            for r in [.05,.2]:
                events.clear()
                llava.generate_llava_baseline(Model(),processor,input_ids=ids,attention_mask=torch.ones_like(ids),
                    pixel_values=torch.zeros(1,3,2,2),image_token_id=99,method=m,retention=r,max_new_tokens=8)
                route.append(dict(method=m,retention=r,events=list(events)))

    env=dict(torch=torch,math=math)
    dart=extract(originals['dart'],'get_retained_image_token',env)
    cos=extract(originals['divprune'],'pairwise_cosine_similarity',dict(torch=torch))
    div=extract(originals['divprune'],'DivPrune',dict(torch=torch))
    obj=SimpleNamespace(pairwise_cosine_similarity=lambda x:cos(None,x))
    comparisons=[]
    for dtype in [torch.float32,torch.float16,torch.bfloat16]:
        for seed in [44,45,46]:
            torch.manual_seed(seed);h=torch.randn(1,610,256).to(dtype);k=torch.randn(1,4,610,64).to(dtype)
            for r in [.05,.2]:
                config=SimpleNamespace(text_length=None,DART_config=dict(K=2,image_token_start_index=4,image_token_length=576,
                    max_num_trunction=None,pivot_image_token=4,pivot_text_token=4,reduction_ratio=1-r))
                a=dart(None,config,h,k);b,_=select_dart(h,k,4,576,round(576*r),torch.nn.Identity())
                f=h[0,4:580];da,_=div(obj,f,576,threshold_ratio=r);db=llava._divprune_select_tokens(f,round(576*r))
                for method,aa,bb in [('dart',a,b),('divprune',da,db)]:
                    aset,bset=set(aa.tolist()),set(bb.tolist())
                    comparisons.append(dict(method=method,dtype=str(dtype),seed=seed,retention=r,
                        author_count=len(aset),current_count=len(bset),intersection=len(aset&bset),
                        jaccard=len(aset&bset)/len(aset|bset),same_order=bool(torch.equal(aa,bb))))
    current={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in [ROOT/'baselines/llava_hf_baselines.py',ROOT/'baselines/llava_dart_corrected.py']}
    result=dict(scope='active shared HF LLaVA entry; not all historical runs; CPU synthetic selector tests, not accuracy',
                references=refs,current_hashes=current,routes=route,selectors=comparisons)
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for m in ['dart','divprune']:
        for dtype in ['torch.float32','torch.float16','torch.bfloat16']:
            rows=[r for r in comparisons if r['method']==m and r['dtype']==dtype]
            print(m,dtype,'same order',sum(r['same_order'] for r in rows),'/',len(rows),
                  'Jaccard',round(min(r['jaccard'] for r in rows),4),round(max(r['jaccard'] for r in rows),4))
    print('entry routes verified:',len(route),'report:',OUT/'audit.json')

if __name__=='__main__':main()
