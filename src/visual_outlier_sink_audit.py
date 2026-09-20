"""Read-only attention reception audit of previously selected hidden17 outliers."""
import json
from pathlib import Path
import torch
from src import causal_effect_benchmark_suite as suite
from src.adapter_single_layer_similarity import Capture
from src.model import load_frozen_qwen3vl,qwen_apply_rotary_pos_emb
from src.qwen_deepstack import disable_qwen_deepstack
from src.data import QwenBenchmarkDataset
ROOT=suite.ROOT
OUT=ROOT/'artifacts/diagnostics/visual_outlier_sink_20260916'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
processor,model=load_frozen_qwen3vl(suite.MODELS['qwen'][0],torch.bfloat16,torch.device('cuda:0'),'flash_attention_2')
disable_qwen_deepstack(model);cap=Capture(model)
old=json.loads((ROOT/'artifacts/diagnostics/initial_token_predictions_l15_17_20260916/hidden17_audit.json').read_text())
results=[]
with torch.inference_mode():
 for benchmark in ['realworldqa','mmstar']:
  path=ROOT/suite.DATASETS[benchmark][0]
  ds=QwenBenchmarkDataset(str(path),processor,benchmark,data_root=str(path.parent),max_samples=suite.DATASETS[benchmark][1])
  for case in [c for c in old['cases'] if c['dataset']==benchmark]:
   item=ds[case['sample']];inputs=suite.prepare(item,model,'qwen');cap.reset();model.model.rope_deltas=None
   model.model(**inputs,use_cache=False,return_dict=True)
   vis=(inputs['mm_token_type_ids'][0]==1).nonzero().flatten()
   text=((inputs['mm_token_type_ids'][0]==0)&(torch.arange(inputs['input_ids'].shape[1],device='cuda')>vis[-1])).nonzero().flatten()
   h=cap.h[17][0,vis].float();norm=h.norm(dim=-1);idx=int(norm.argmax());pos=int(vis[idx])
   result={'dataset':benchmark,'sample':case['sample'],'visual_tokens':len(vis),'text_queries':len(text),'outlier_visual_index':idx,'norm_ratio':float(norm.max()/norm.median()),'layers':{}}
   for l in range(15,23):
    attn=cap.layers[l].self_attn;n=cap.norm[l];shape=(*n.shape[:-1],-1,attn.head_dim)
    q=attn.q_norm(attn.q_proj(n).view(shape)).transpose(1,2)
    k=attn.k_norm(attn.k_proj(n).view(shape)).transpose(1,2)
    q,k=qwen_apply_rotary_pos_emb(q,k,*cap.rope[l]);k=k.repeat_interleave(q.shape[1]//k.shape[1],1)
    # All post-image text queries; full causal denominator includes text and image.
    logits=(q[0,:,text].float()@k[0].float().transpose(-1,-2))*float(attn.scaling)
    keys=torch.arange(n.shape[1],device='cuda');logits.masked_fill_(keys[None,None,:]>text[None,:,None],-torch.inf)
    a=logits.softmax(-1);visual_a=a[:,:,vis]
    incoming=visual_a.mean((0,1));full=float(incoming[idx]);total=float(incoming.sum())
    conditional=visual_a/visual_a.sum(-1,keepdim=True).clamp_min(1e-20)
    perhead=conditional[:,:,idx].mean(-1)
    top=int(incoming.argmax())
    result['layers'][str(l)]={'outlier_full_attention':full,'all_visual_attention':total,'outlier_share_of_visual_mass':full/total,'outlier_visual_attention_rank':int((incoming>incoming[idx]).sum())+1,'outlier_mean_query_conditional_attention':float(conditional[:,:,idx].mean()),'outlier_max_head_mean_conditional_attention':float(perhead.max()),'heads_above_50pct':int((perhead>.5).sum()),'heads_above_10pct':int((perhead>.1).sum()),'top_visual_index':top,'top_visual_full_attention':float(incoming[top]),'last_query_outlier_full_attention':float(a[:,-1,pos].mean()),'norm_ratio_at_this_layer':float(cap.h[l][0,vis[idx]].float().norm()/cap.h[l][0,vis].float().norm(dim=-1).median())}
   results.append(result);(OUT/'results.json').write_text(json.dumps({'cases':results,'protocol':'Native BF16 FA2, DeepStack off. FP32 offline QK softmax, original full causal denominator. Post-image text queries, all32 heads. Same native max-norm token at layer17 tracked at layers15–22. Six selected cases, not population prevalence.'},indent=2))
   print(benchmark,case['sample'],'ratio',result['norm_ratio'], {l:(round(v['outlier_full_attention'],4),round(v['outlier_share_of_visual_mass'],4),v['outlier_visual_attention_rank']) for l,v in result['layers'].items()},flush=True)
cap.close()
