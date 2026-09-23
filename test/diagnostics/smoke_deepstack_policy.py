import json
from pathlib import Path
import torch
from PIL import Image
from src.model import load_frozen_qwen3vl
processor,model=load_frozen_qwen3vl('/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct',torch.bfloat16,torch.device('cuda:0'))
counts={'main_merger':0,'deepstack_merger':0,'language_injection':0}
def bump(name):
 def hook(*a):counts[name]+=1
 return hook
model.model.visual.merger.register_forward_hook(bump('main_merger'))
for module in model.model.visual.deepstack_merger_list:module.register_forward_hook(bump('deepstack_merger'))
original=model.model.language_model._deepstack_process
def guard(*a,**kw):counts['language_injection']+=1;return original(*a,**kw)
model.model.language_model._deepstack_process=guard
rows=[]
with torch.inference_mode():
 for name,images in [('single_image',[Image.new('RGB',(64,64),'red')]),('multi_image',[Image.new('RGB',(64,64),'red'),Image.new('RGB',(64,64),'blue')])]:
  messages=[{'role':'user','content':[{'type':'image'} for _ in images]+[{'type':'text','text':'Describe the image briefly.'}]}]
  text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  batch=processor(text=[text],images=images,return_tensors='pt').to('cuda:0')
  result=model.generate(**batch,max_new_tokens=2,do_sample=False)
  rows.append({'input':name,'generated_tokens':result.shape[1]-batch['input_ids'].shape[1]})
 # Direct video features exercises the same vision branch with temporal input.
 cfg=model.config.vision_config
 grid=torch.tensor([[2,4,4]],device='cuda:0')
 pixels=torch.randn(32,cfg.in_channels*cfg.temporal_patch_size*cfg.patch_size**2,device='cuda:0',dtype=torch.bfloat16)
 features=model.model.visual(pixels,grid_thw=grid)
 assert features.deepstack_features==[]
assert counts=={'main_merger':3,'deepstack_merger':0,'language_injection':0},counts
out={'passed':True,'attention':model.config.text_config._attn_implementation,'counts':counts,'generation':rows,'video_features':'passed','teacher_frozen':not any(p.requires_grad for p in model.parameters())}
Path('artifacts/diagnostics/deepstack_policy_20260920').mkdir(exist_ok=True,parents=True)
Path('artifacts/diagnostics/deepstack_policy_20260920/smoke.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
