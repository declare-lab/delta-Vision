"""Functional transfer of paired MSE adapters under identical native text queries."""
import json,torch
from pathlib import Path
import torch.nn.functional as F
from src.initial_token_mlp_probe import runtime,teacher
from src.model import qwen_apply_rotary_pos_emb,load_qwen_embedding_adapter_checkpoint
from src.adapter_single_layer_similarity import projected_kv
from src.data import QwenBenchmarkDataset
from src.visual_channel_rank_grid import _to_device_item
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'artifacts/experiments/adapter_mse_fourway_20260916';OUT=ROOT/'artifacts/diagnostics/mse_fixed_q_20260916';OUT.mkdir(exist_ok=True)
LS=(13,15,16,17,18,20,22,29,33)
runtime();processor,model=teacher();adapters={}
paths={name:EXP/name/'checkpoints/qwen_embedding_adapter_step2000.pt' for name in ('pre_uniform','pre_norm3','post_uniform')}
paths['kl_reference']=ROOT/'artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt'
for name,path in paths.items():
 adapters[name],meta=load_qwen_embedding_adapter_checkpoint(path,model.model.language_model,model.device,torch.bfloat16);assert not meta['missing'] and not meta['unexpected']
h={};z={};rope={};handles=[];layers=model.model.language_model.layers
for l in (0,*LS):
 def hook(m,args,kw,l=l):h[l]=kw.get('hidden_states',args[0] if args else None).detach()
 handles.append(layers[l].register_forward_pre_hook(hook,with_kwargs=True))
for l in LS:
 def hook(m,args,kw,l=l):z[l]=kw['hidden_states'].detach();rope[l]=kw['position_embeddings']
 handles.append(layers[l].self_attn.register_forward_pre_hook(hook,with_kwargs=True))
def js(p,q):
 m=(p+q)/2
 return .5*((p*(p.clamp_min(1e-30).log2()-m.clamp_min(1e-30).log2())).sum(-1)+(q*(q.clamp_min(1e-30).log2()-m.clamp_min(1e-30).log2())).sum(-1))
with torch.no_grad(),(OUT/'rows.jsonl').open('w',buffering=1) as log:
 for b in ('realworldqa','mmstar','sqa'):
  ds=QwenBenchmarkDataset(str(ROOT/f'artifacts/diagnostics/channel_native_cache_20260916/{b}_eval.jsonl'),processor,b)
  for i in sorted(set(round(j*(len(ds)-1)/63) for j in range(64))):
   inputs=_to_device_item(ds[i],model.device);inputs={k:v for k,v in inputs.items() if k in ('input_ids','attention_mask','pixel_values','image_grid_thw','mm_token_type_ids')}
   h.clear();z.clear();rope.clear();model.model.rope_deltas=None;model.model(**inputs,use_cache=False)
   vis=(inputs['mm_token_type_ids'][0]==1).nonzero().flatten();text=((inputs['mm_token_type_ids'][0]==0)&(torch.arange(inputs['input_ids'].shape[1],device='cuda')>vis[-1])).nonzero().flatten();E=h[0][:,vis]
   memories={n:a.all_visual_memories_batched(E) for n,a in adapters.items()};metrics={}
   for l in LS:
    layer=layers[l];a=layer.self_attn;shape=(*z[l].shape[:-1],-1,a.head_dim)
    q=a.q_norm(a.q_proj(z[l]).view(shape)).transpose(1,2);k=a.k_norm(a.k_proj(z[l]).view(shape)).transpose(1,2);v=a.v_proj(z[l]).view(shape).transpose(1,2)
    q,k=qwen_apply_rotary_pos_emb(q,k,*rope[l]);q=q[0,:,text].float();groups=q.shape[0]//k.shape[1];kv=k[0,:,vis].repeat_interleave(groups,0).float();vv=v[0,:,vis].repeat_interleave(groups,0).float()
    nq= q@kv.transpose(-1,-2)*float(a.scaling);np=nq.softmax(-1)
    raw=h[l][0,vis].float();norm=raw.norm(dim=-1);outlier=norm>3*norm.median();ordinary=~outlier
    # Conditional-on-visual distributions. All selected text queries follow all visual tokens.
    # Text key contribution to full softmax, same for every adapter.
    keys=torch.arange(z[l].shape[1],device='cuda');fullk=k[0].repeat_interleave(groups,0).float();base_logits=q@fullk.transpose(-1,-2)*float(a.scaling);base_logits.masked_fill_(keys[None,None,:]>text[None,:,None],-torch.inf)
    textkey=torch.ones(len(keys),device='cuda',dtype=torch.bool);textkey[vis]=False;tlse=base_logits[:,:,textkey].logsumexp(-1)
    native_mass=torch.sigmoid(nq.logsumexp(-1)-tlse);metrics[str(l)]={}
    for name,mem in memories.items():
     ak,av=projected_kv(layer,mem[l],rope[l],vis);ak=ak[0].repeat_interleave(groups,0).float();av=av[0].repeat_interleave(groups,0).float();sq=q@ak.transpose(-1,-2)*float(a.scaling);sp=sq.softmax(-1)
     restored=sq.clone();restored[:,:,outlier]=nq[:,:,outlier];rp=restored.softmax(-1)
     restoreordinary=sq.clone();restoreordinary[:,:,ordinary]=nq[:,:,ordinary];rop=restoreordinary.softmax(-1)
     studentmass=torch.sigmoid(sq.logsumexp(-1)-tlse)
     prednorm=layer.input_layernorm(mem[l])[0].float();targetnorm=z[l][0,vis].float()
     metrics[str(l)][name]={'visual_attention_js':float(js(np,sp).mean()),'visual_attention_cosine':float(F.cosine_similarity(np,sp,dim=-1).mean()),'js_restore_outlier_keys':float(js(np,rp).mean()),'js_restore_ordinary_keys':float(js(np,rop).mean()),'visual_mass_abs_error':float((native_mass-studentmass).abs().mean()),'native_outlier_conditional_mass':float(np[:,:,outlier].sum(-1).mean()),'student_outlier_conditional_mass':float(sp[:,:,outlier].sum(-1).mean()),'ordinary_postnorm_mse':float((prednorm[ordinary]-targetnorm[ordinary]).square().mean()),'ordinary_key_mse':float((ak[:,ordinary]-kv[:,ordinary]).square().mean()),'ordinary_value_mse':float((av[:,ordinary]-vv[:,ordinary]).square().mean())}
   log.write(json.dumps({'benchmark':b,'sample':i,'metrics':metrics})+'\n')
   if i==0 or i>len(ds)-20:print(b,i,flush=True)
rows=[json.loads(x) for x in (OUT/'rows.jsonl').read_text().splitlines()];summary={}
for b in ('realworldqa','mmstar','sqa'):
 rr=[r for r in rows if r['benchmark']==b];assert len(rr)==64;summary[b]={}
 for l in LS:
  summary[b][str(l)]={name:{k:sum(r['metrics'][str(l)][name][k] for r in rr)/64 for k in rr[0]['metrics'][str(l)][name]} for name in paths}
(OUT/'results.json').write_text(json.dumps(summary,indent=2));(OUT/'status.json').write_text(json.dumps({'state':'complete','samples':192,'selection':'64 evenly spaced indices per benchmark','query':'frozen native text Q, all post-image queries and32heads','restore':'offline keys only, fixed native Q and textK; no autoregressive accuracy intervention'}))
