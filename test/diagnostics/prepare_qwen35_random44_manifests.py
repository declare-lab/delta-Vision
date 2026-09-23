"""Freeze uniform seed44 question subsets for the corrected six-method comparison."""
import argparse,collections,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.evaluation_sampling import sample_evaluation_rows


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser();p.add_argument('--reference-run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 cfg=json.loads((a.reference_run/'config.json').read_text())
 a.output_dir.mkdir(parents=True,exist_ok=False)
 metadata=dict(seed=44,sampling='uniform_without_replacement',max_questions=1000,
               methods=['native','adapter','divprune_5','divprune_20','dart_5','dart_20'],benchmarks={})
 for name,info in cfg['evaluation'].items():
  path=Path(info['source']);all_rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
  rows,indices=sample_evaluation_rows(all_rows,limit=1000,seed=44)
  dest=a.output_dir/f'{name}.jsonl';dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
  entry=dict(source=str(path),source_sha256=digest(path),path=str(dest.resolve()),sha256=digest(dest),
             image_root=info['image_root'],samples=len(rows),source_indices=indices,
             categories=dict(collections.Counter(r.get('category','unspecified') for r in rows)))
  metadata['benchmarks'][name]=entry
  print(name,len(rows),entry['categories'])
 (a.output_dir/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
