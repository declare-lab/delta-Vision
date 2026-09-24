"""Compare migrated numerical function/class ASTs against saved original sources."""
import ast,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'artifacts/maintenance/paper_analysis_layout_20260924';m=json.loads((base/'migration.json').read_text());ref=base/'reference'
mods={v[:-3].replace('/','.'):k[:-3].replace('/','.') for k,v in m['mapping'].items()}
mods['analysis.fig03_visual_effect.native_effect_core']='src.visual_effect_svd'
class Normalize(ast.NodeTransformer):
 def visit_ImportFrom(self,n):
  return None
 def visit_Import(self,n):return None
 def visit_Constant(self,n):
  if isinstance(n.value,str):
   s=n.value
   for a,b in mods.items():s=s.replace(a,b)
   for a,b in ((v,k) for k,v in m['mapping'].items()):s=s.replace(a,b)
   n.value=s
  return n
pairs={**m['mapping'],**{p:p for p in m['recovered']}}
report={};changed=[]
for old,new in pairs.items():
 a=ast.parse((ref/old).read_text());b=ast.parse((ROOT/new).read_text());left={n.name:n for n in a.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};right={n.name:n for n in b.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 same=[];different=[]
 for name,x in left.items():
  y=right[name]
  if ast.dump(Normalize().visit(x))==ast.dump(Normalize().visit(y)):same.append(name)
  else:different.append(name)
 report[new]={'unchanged_numeric_or_function_bodies':same,'changed_bodies_review_required':different}
 if different:changed.append((new,different))
(base/'ast_inventory.json').write_text(json.dumps(report,indent=2)+'\n')
print('Functions/classes unchanged:',sum(len(x['unchanged_numeric_or_function_bodies']) for x in report.values()));print('Changed:',changed)

for path,names in changed:
 allowed={'prepare'}
 if path=='analysis/fig04_adapter_rank/eval_pixmo_static_rank_sweep.py':allowed.add('main')
 assert set(names)<=allowed,(path,names)
print('PASS: only snapshot preparation and the relocated worker launch differ beyond imports/paths')
