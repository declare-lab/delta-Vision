"""Render old and new plot implementations in temporary output directories."""
import ast,hashlib,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];AUDIT=ROOT/'artifacts/maintenance/paper_analysis_layout_20260924'
source=ROOT/'artifacts/diagnostics/initial_token_postnorm_all36_20260916/results.csv'
paths={'old':AUDIT/'reference/scripts/plot_layerwise_residual_mlp.py','new':ROOT/'analysis/fig01b_hidden_prediction/plot_layerwise_residual_mlp.py'}
with tempfile.TemporaryDirectory(prefix='analysis_plot_') as temp:
 generated={}
 for side,path in paths.items():
  tree=ast.parse(path.read_text());out=Path(temp)/side
  for n in tree.body:
   if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('SOURCE','OUT'):
    target=source if n.targets[0].id=='SOURCE' else out
    n.value=ast.parse('Path('+repr(str(target))+')',mode='eval').body
  ast.fix_missing_locations(tree);exec(compile(tree,str(path),'exec'),{'__file__':str(path),'__name__':'__main__'})
  generated[side]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.png')}
 assert generated['old'] and generated['old']==generated['new'],generated
 (AUDIT/'plot_comparison.json').write_text(json.dumps({'png_sha256':generated['new'],'all_png_bitwise_equal':True,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'note':'Only SOURCE/OUT assignments redirected to avoid modifying historical artifacts. PDFs have timestamps; compared rendered PNGs.'},indent=2)+'\n')
 print('Identical PNGs:',len(generated['new']))
