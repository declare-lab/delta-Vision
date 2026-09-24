"""Aggregate per-token/head norm sums (never norm-of-sum or clipped C)."""
import argparse
import json
from pathlib import Path
import numpy as np


def aggregate(rows,kind):
    data=[r[kind] for r in rows if r[kind]['positions']]
    a=np.asarray([r['per_head_sums'] for r in data]).sum(0)
    n=sum(r['positions'] for r in data)
    total=a.sum(0)
    return dict(cancellation_ratio=float(1-total[1]/total[0]),
        cancellation_per_head=(1-a[:,1]/a[:,0]).tolist(),
        raw_write_norm_sum=float(total[0]),delta_write_norm_sum=float(total[1]),
        raw_write_norm_mean=float(total[0]/n/32),delta_write_norm_mean=float(total[1]/n/32),
        value_norm_mean=float(total[2]/n/32),residual_norm_mean=float(total[3]/n/32),
        predicted_norm_mean=float(total[4]/n/32),
        value_prediction_cosine=float(total[5]/np.sqrt(total[6]*total[7])) if total[7]>0 else None,
        forgetting_norm_mean=float(total[8]/n/32),total_step_norm_mean=float(total[9]/n/32),
        c_gt_09_fraction=sum(r['c_gt_09_count'] for r in data)/n/32,
        negative_c_fraction=sum(r['negative_c_count'] for r in data)/n/32,
        example_mean_c=float(np.mean([r['cancellation_ratio'] for r in data])),positions=n)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--partial',action='store_true')
    args=p.parse_args();run=args.run;config=json.loads((run/'config.json').read_text())
    report=dict(complete=not args.partial,scope=config['numeric'],benchmarks={})
    layers=list(i for i in range(32) if i%4!=3)
    for name,info in config['evaluation'].items():
        rows=[json.loads(l) for f in (run/'results').glob(f'{name}.shard*.jsonl') for l in f.read_text().splitlines()]
        if not rows:continue
        assert len({r['index'] for r in rows})==len(rows)
        if not args.partial:assert len(rows)==info['samples']
        methods=['native','visual_beta_zero','no_cancel','replay_control','no_cancel_pure']
        stats={m:dict(accuracy=100*np.mean([r['variants'][m]['score'] for r in rows]),
            invalid=sum(r['variants'][m]['invalid'] for r in rows),
            changed_tokens_vs_native=sum(r['variants'][m]['generated_token_ids']!=r['variants']['native']['generated_token_ids'] for r in rows)) for m in methods}
        for m in methods[1:]:
            kl=[r['variants'][m]['kl_to_native'] for r in rows]
            stats[m].update(mean_kl=float(np.mean(kl)),max_kl=max(kl),
                mean_text_hidden_relative_error={str(i):float(np.mean([r['variants'][m]['text_hidden_relative_error'][str(i)] for r in rows])) for i in range(32)})
        layer_records={i:[next(x for x in r['layers'] if x['layer']==i) for r in rows] for i in layers}
        report['benchmarks'][name]=dict(samples=len(rows),accuracy=stats,
            native_generation_exact=all(r['native_generation_exact'] for r in rows),
            max_replay_core_relative_error=max(x['checks']['replay_vs_native_core'] for r in rows for x in r['layers']),
            layers=[dict(layer=i,**{k:aggregate(layer_records[i],k) for k in ('visual','text','text_before','text_after')}) for i in layers])
    if not args.partial:assert len(report['benchmarks'])==3
    dest=run/('partial' if args.partial else 'reports');dest.mkdir(exist_ok=True)
    (dest/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# GDN visual delta correction','',
        'Native Qwen3.5-4B; DeepStack off; FA2 full attention. Fixed seed44 manifests. SQA/MMStar 1000 each, RWQA765.',
        'Generation: original max8, greedy, thinking disabled; unfinished responses score zero. Prefill-only statistics; first-answer full-vocabulary KL.',
        'C = 1 - sum(token,head) ||delta write|| / sum(token,head) ||raw write||. Negative values retained. C measures delta write, not forgetting or total state change.',
        'Statistics use an FP32 algebraic replay of native post-convolution inputs and BF16-normalized Q/K; numerical differences from native chunk execution are separately measured.',
        'No-cancel uses native + (changed replay - native replay) per layer; text decode uses native recurrence with the changed cache. Replay control reports approximation sensitivity.',
        '', '| Setting | SQA | RealWorldQA | MMStar | Avg. |','|---|---:|---:|---:|---:|']
    for key,label in [('native','Native'),('visual_beta_zero','Visual beta=0 (decay retained)'),('no_cancel','Visual subtraction removed (paired)'),('replay_control','FP32 replay numerical control'),('no_cancel_pure','Visual subtraction removed (pure replay)')]:
        vals=[report['benchmarks'].get(b,{}).get('accuracy',{}).get(key,{}).get('accuracy') for b in ('sqa','realworldqa','mmstar')]
        avg=np.mean(vals) if all(v is not None for v in vals) else None
        lines.append('| '+label+' | '+' | '.join('—' if x is None else f'{x:.2f}' for x in vals+[avg])+' |')
    for name,b in report['benchmarks'].items():
        lines+=['',f'## {name}: {b["samples"]} examples','',
            f'Native generated tokens exactly match reference: {b["native_generation_exact"]}. Maximum core replay relative error: {b["max_replay_core_relative_error"]:.6f}.',
            f'Replay-control changed generations: {b["accuracy"]["replay_control"]["changed_tokens_vs_native"]}; mean KL: {b["accuracy"]["replay_control"]["mean_kl"]:.6g}.',
            f'No-cancel mean KL: {b["accuracy"]["no_cancel"]["mean_kl"]:.6g}.',
            '', '| Layer | Visual C | Text C | Visual value norm | Visual residual norm | Visual fraction C>0.9 |','|---:|---:|---:|---:|---:|---:|']
        for x in b['layers']:
            v=x['visual'];t=x['text']
            lines.append(f'| {x["layer"]} | {v["cancellation_ratio"]:.5f} | {t["cancellation_ratio"]:.5f} | {v["value_norm_mean"]:.5g} | {v["residual_norm_mean"]:.5g} | {v["c_gt_09_fraction"]:.4%} |')
    (dest/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for norm in (False,True):
        fig,axs=plt.subplots(1,4,figsize=(15,3.3),sharex=True)
        curves=[]
        for ax,(name,b) in zip(axs,report['benchmarks'].items()):
            a=np.array([[x['visual']['value_norm_mean'],x['visual']['residual_norm_mean']] if norm else [x['visual']['cancellation_ratio'],x['text']['cancellation_ratio']] for x in b['layers']])
            curves.append(a)
            for j in range(2):ax.plot(layers,a[:,j],marker='o',markersize=2,label=(['Value norm','Residual norm'] if norm else ['Visual','Text'])[j])
            ax.set_title(f'{name} (n={b["samples"]})')
        if curves:
            a=np.mean(curves,axis=0)
            for j in range(2):axs[-1].plot(layers,a[:,j],marker='o',markersize=2,label=(['Value norm','Residual norm'] if norm else ['Visual','Text'])[j])
        axs[-1].set_title('Dataset macro average')
        for ax in axs:
            ax.set_xlabel('GDN layer (0-based)');ax.grid(alpha=.2)
            if norm:ax.set_yscale('log')
            else:ax.axhline(0,color='gray',linewidth=.7)
        axs[0].set_ylabel('Mean norm per token/head' if norm else 'Cancellation ratio C')
        axs[-1].legend(fontsize=8);fig.tight_layout()
        name='value_residual_norm' if norm else 'cancellation_ratio'
        for suffix in ('pdf','png'):fig.savefig(dest/f'{name}.{suffix}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    print(dest/'RESULTS.md')


if __name__=='__main__':main()
