"""Compare DivPrune selectors on actual checkpoint visual features, not accuracy."""
import json
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace
import torch
from transformers import AutoConfig,AutoProcessor,CLIPVisionModel
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
from safetensors import safe_open
from src.data import LlavaBenchmarkDataset
from baselines.llava_hf_baselines import _divprune_select_tokens

def main():
    root=Path(__file__).resolve().parents[2]
    extract=runpy.run_path(str(root/'test/diagnostics/audit_llava_six_routes.py'))['extract']
    p=Path('/lustre-data/leijingdi/code/delta-vision/models/llava-1.5-7b-hf')
    torch.set_num_threads(4)
    c=AutoConfig.from_pretrained(p);c.vision_config._attn_implementation='flash_attention_2'
    v=CLIPVisionModel(c.vision_config).eval();projector=LlavaMultiModalProjector(c).eval()
    index=json.loads((p/'model.safetensors.index.json').read_text())['weight_map']
    states={'vision_tower':{},'multi_modal_projector':{}}
    for filename in set(index.values()):
        keys=[k for k in index if index[k]==filename and any(k.startswith(prefix+'.') for prefix in states)]
        if not keys:continue
        with safe_open(p/filename,framework='pt',device='cpu') as f:
            for k in keys:
                prefix,k2=k.split('.',1);states[prefix][k2]=f.get_tensor(k)
    # HF 5 flattened CLIPVisionModel. Match the checkpoint-conversion mapping.
    vision_state=states['vision_tower']
    if 'embeddings.class_embedding' in v.state_dict():
        vision_state={k.removeprefix('vision_model.'):t for k,t in vision_state.items()}
    v.load_state_dict(vision_state);projector.load_state_dict(states['multi_modal_projector']);del states,vision_state
    v=v.to('cuda',torch.bfloat16);projector=projector.to('cuda',torch.bfloat16)
    processor=AutoProcessor.from_pretrained(p)
    raw=subprocess.check_output(['git','-C',str(root/'baselines/divprune'),'show','HEAD:LLaVA/llava/model/llava_arch.py'],text=True)
    cos=extract(raw,'pairwise_cosine_similarity',dict(torch=torch));select=extract(raw,'DivPrune',dict(torch=torch))
    obj=SimpleNamespace(pairwise_cosine_similarity=lambda x:cos(None,x))
    cfg=json.loads((root/'artifacts/eval/native_initial_visual_random44_20260921/config.json').read_text());rows=[]
    with torch.inference_mode():
        for b in ['mmstar','realworldqa','sqa']:
            info=cfg['evaluation'][b];ds=LlavaBenchmarkDataset(info['path'],processor,b,data_root=info['image_root'])
            item=ds[0]
            features=v(item['pixel_values'][None].cuda().to(torch.bfloat16),output_hidden_states=True)
            f=projector(features.hidden_states[c.vision_feature_layer][:,1:])[0]
            for r in [.05,.2]:
                cur=_divprune_select_tokens(f,round(len(f)*r));cs=set(cur.tolist())
                for dtype in [torch.float32,torch.float16,torch.bfloat16]:
                    old,_=select(obj,f.to(dtype),len(f),threshold_ratio=r);a=set(old.tolist())
                    rec=dict(model='llava-1.5-7b',benchmark=b,sample=0,retention=r,
                        author_distance_dtype=str(dtype),current_distance_dtype='float32',
                        count=len(cs),intersection=len(a&cs),jaccard=len(a&cs)/len(a|cs),same_order=bool(torch.equal(old,cur)))
                    rows.append(rec);print(rec,flush=True)
    out=root/'artifacts/reports/llava_six_author_audit_20260922/real_feature_divprune.json'
    out.write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
