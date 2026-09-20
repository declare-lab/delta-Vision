"""Compare late-start checkpoints on the frozen, matched MuirBench 1000 rows."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'artifacts/diagnostics/muir_random1000_seed42_matched_20260914'
NEW = ROOT/'artifacts/diagnostics/muir_late_adapter_random1000_20260914'
METHODS = {
    'base': ('Base', OLD),
    'embedding_adapter': ('Embedding Adapter: start0, rank128', OLD),
    'embedding_adapter_mixed': ('Mixed-data Adapter: start0, rank128', OLD),
    'adapter_start7': ('Embedding Adapter: start7, rank512', NEW),
    'adapter_start16': ('Embedding Adapter: start16, rank512', NEW),
    'dart': ('DART 20%', OLD),
}


def main():
    rows = {}
    for method, (_, directory) in METHODS.items():
        selected = [json.loads(line) for p in directory.glob(f'{method}_shard*.jsonl')
                    for line in p.open() if line.strip()]
        selected = [r for r in selected if r['benchmark']=='muirbench'
                    and r['retention']==(.2 if method=='dart' else 1.)]
        assert len(selected)==1000 and {r['index'] for r in selected}==set(range(1000)), (method,len(selected))
        rows[method] = {r['index']:r for r in selected}
    for method, values in rows.items():
        for i, r in values.items():
            base = rows['base'][i]
            for key in ('input_sha256','dataset_sha256','source_index','prompt_layout','max_new_tokens'):
                assert r[key]==base[key], (method,i,key)
            assert r['deepstack_enabled'] is False
    summary = {m:dict(n=1000,correct=sum(r['score'] for r in rs.values()),
                     accuracy=sum(r['score'] for r in rs.values())/10,
                     invalid=sum(not r['prediction'] for r in rs.values())) for m,rs in rows.items()}
    paired = {}
    for method in ('adapter_start7','adapter_start16'):
        paired[method] = dict(Counter(
            f"{int(rows['embedding_adapter'][i]['score'])}->{int(r['score'])}"
            for i,r in rows[method].items()))
    tasks = sorted({r['task'] for r in rows['base'].values()})
    task_results = {}
    for task in tasks:
        ids=[i for i,r in rows['base'].items() if r['task']==task]
        task_results[task] = dict(n=len(ids), **{
            m:100*sum(rs[i]['score'] for i in ids)/len(ids) for m,rs in rows.items()})
    result=dict(overall=summary,paired_vs_start0=paired,tasks=task_results,
                exact_input_hash_match=True,deepstack=False)
    (NEW/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# MuirBench: native prefix followed by embedding adapter','',
           'Frozen random 1000 rows, seed 42; all images in original order/resolution; '
           'media_first_v1; no DeepStack; BF16; greedy ≤8 tokens; identical scorer. '
           'All 1000 processed-input hashes match the existing Base/Adapter/DART evaluation.','',
           '| Model | Correct / 1000 | Accuracy |','|---|---:|---:|']
    for m,(label,_) in METHODS.items():
        s=summary[m];lines.append(f"| {label} | {s['correct']:.0f} | {s['accuracy']:.2f}% |")
    lines += ['', 'Layer numbers are zero-based. start7: native 0–6 → adapter 7–34 → native 35. '
              'start16: native 0–15 → adapter 16–34 → native 35. Both adapters use the '
              'native-prefix rollout as the fixed anchor; layer 35 receives the last adapted visual memory '
              'without an extra boundary FFN.', '',
              'These are existing checkpoint comparisons, NOT a controlled start-layer-only ablation: '
              'late-start rank512 differs from current start0 rank128, and the native tail differs too. '
              'Any improvement alone cannot isolate cross-image context as the cause.', '',
              '| Task | N | Base | start0 | mixed | start7 | start16 | DART20 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for task,values in task_results.items():
        lines.append('| '+task+' | '+str(values['n'])+' | '+' | '.join(f'{values[m]:.2f}' for m in METHODS)+' |')
    (NEW/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['overall'],indent=2))


if __name__=='__main__':
    main()
