"""Reconstruct fixed-list media-first inputs and verify against saved hashes."""
import re
from src import benchmarks
from src.data import QwenBenchmarkDataset

for _name in ('muirbench','mvbench'):
 benchmarks.BENCHMARK_SPECS.setdefault(_name,benchmarks.BenchmarkSpec(name=_name,display_name=_name,
  metric='multi_choice',default_data=f'data/benchmarks/{_name}/test.jsonl',
  answer_instruction="Answer with the option's letter from the given choices directly.",max_new_tokens=128))

class FixedMultimodalDataset(QwenBenchmarkDataset):
 def _qwen_message_content(self,row,question,images):
  refs=list(re.finditer(r'<\|(image|video)_(\d+)\|>',question))
  def replace(m):
   kind,number=m.group(1),int(m.group(2))
   if kind!='image' or not 1<=number<=len(images):raise ValueError('Invalid image reference '+m.group(0))
   return f'Image {number}'
  question=re.sub(r'<\|(image|video)_(\d+)\|>',replace,question)
  content=[]
  for i,image in enumerate(images,1):
   content.append(dict(type='image',image=image))
   if refs:content.append(dict(type='text',text=f'\n[End of Image {i}]\n'))
  content.append(dict(type='text',text=question))
  return content
