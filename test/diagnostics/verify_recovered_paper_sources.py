"""Compare recovered original algorithms, gradients, and complete historical rank aggregates."""
import contextlib,hashlib,importlib.util,io,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(2)
BASE=ROOT/'artifacts/maintenance/paper_source_recovery_20260924'

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
rank_old=load('rank_original',BASE/'reference/src/visual_rank_statistics.py')
obj_old=load('objective_original',BASE/'reference/src/pixmo_objective_comparison.py')
from analysis.table05_native_rank import visual_rank_statistics as rank_new
from analysis.table10_training_objective import pixmo_objective_comparison as obj_new
results={};torch.manual_seed(44)
q,k=torch.randn(2,17,8,dtype=torch.float64),torch.randn(2,17,8,dtype=torch.float64)
for name in ['score_and_query_spectra','feature_singular_values','spectral_metrics']:
 args=(q,k,.25) if name=='score_and_query_spectra' else (q[0],) if name=='feature_singular_values' else (torch.tensor([4.,1.,0.],dtype=torch.float64),)
 a=getattr(rank_old,name)(*args);b=getattr(rank_new,name)(*args)
 pairs=zip(a,b) if isinstance(a,tuple) else [(a,b)]
 assert all(torch.equal(x,y) for x,y in pairs);results[name]='bitwise_equal'
s=torch.randn(19,137,requires_grad=True);t=torch.randn_like(s)
a=obj_old.FullVocabKL.apply(s,t,2.,7);ga=torch.autograd.grad(a,s)[0]
b=obj_new.FullVocabKL.apply(s,t,2.,7);gb=torch.autograd.grad(b,s)[0]
assert torch.equal(a,b) and torch.equal(ga,gb)
results['kl_loss_max_abs']=float((a-b).abs().detach());results['kl_gradient_max_abs']=float((ga-gb).abs().max())
source=ROOT/'artifacts/diagnostics/native_visual_rank_20260912'
expected=json.loads((source/'provenance.json').read_text())['source_sha256']['src/visual_rank_statistics.py']
assert hashlib.sha256((BASE/'reference/src/visual_rank_statistics.py').read_bytes()).hexdigest()==expected
with tempfile.TemporaryDirectory(prefix='recovered_rank_merge_') as temp:
 outputs={}
 for label,module in [('old',rank_old),('new',rank_new)]:
  folder=Path(temp)/label;folder.mkdir()
  for shard in source.glob('*_shard[0-9]*.jsonl'):(folder/shard.name).symlink_to(shard)
  with contextlib.redirect_stdout(io.StringIO()):module.merge(SimpleNamespace(output=str(folder),limit=0))
  outputs[label]={name:json.loads((folder/f'{name}.json').read_text()) for name in ['overall','per_layer','per_head']}
 assert outputs['old']==outputs['new']
 for name,rows in outputs['new'].items():assert rows==json.loads((source/f'{name}.json').read_text()),name
 results['historical_aggregation']={name:{'rows':len(rows),'exact_match':True} for name,rows in outputs['new'].items()}
(BASE/'validation.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
