"""Aggregate complete probe shards and produce publication-exportable figures."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics


def numeric_leaves(value,prefix=''):
    for key,v in value.items():
        path=f'{prefix}.{key}' if prefix else key
        if isinstance(v,dict):yield from numeric_leaves(v,path)
        elif isinstance(v,(int,float)):yield path,float(v)
        elif isinstance(v,list) and all(isinstance(x,(int,float)) for x in v):
            for i,x in enumerate(v):yield f'{path}[{i}]',float(x)


def summarize_groups(records,keys):
    values=defaultdict(lambda:defaultdict(list))
    counts=defaultdict(int)
    for row in records:
        group=tuple(row[k] for k in keys);counts[group]+=1
        for key,value in numeric_leaves(row):
            if key not in keys: values[group][key].append(value)
    return [dict(zip(keys,g),samples=counts[g],means={k:statistics.mean(v) for k,v in data.items()}) for g,data in values.items()]


def load(run,stage,config):
    records=[];complete=True
    for benchmark,info in config['evaluation'].items():
        rows=[json.loads(line) for p in sorted((run/stage).glob(f'{benchmark}.shard*.jsonl')) for line in p.read_text().splitlines()]
        assert len({r['index'] for r in rows})==len(rows)
        if len(rows)!=info['samples']:complete=False
        records.extend(rows)
    return records,complete


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();run=args.run
    config=json.loads((run/'config.json').read_text());dest=run/'reports';dest.mkdir(exist_ok=True)
    expected=sum(v['samples'] for v in config['evaluation'].values())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    benchmarks=list(config['evaluation']);titles={'sqa':'SQA','realworldqa':'RealWorldQA','mmstar':'MMStar','macro':'Dataset macro mean'}
    report=['# Qwen3.5 memory mechanism results','',
        'Only complete stages produce figures/tables. Every head is decomposed independently; r95 uses squared singular-value energy. Macro curves equally weight datasets. See config.json and the project-root README.md for controls.','']
    def save(fig,name):
        fig.savefig(dest/f'{name}.pdf',bbox_inches='tight');fig.savefig(dest/f'{name}.png',dpi=180,bbox_inches='tight');plt.close(fig)
    analysis,complete=load(run,'analysis',config)
    report.append(f'Analysis: {len(analysis)}/{expected} samples; complete={complete}.')
    if complete:
        flat=[dict(benchmark=r['benchmark'],**x) for r in analysis for x in r['layers']]
        summary=summarize_groups(flat,['benchmark','method','layer','block_type'])
        (dest/'layer_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
        lookup={(r['benchmark'],r['method'],r['layer']):r['means'] for r in summary}
        with (dest/'state_head_rank_means.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['benchmark','method','layer_0based','head_0based','r90','r95','effective_rank'])
            for r in summary:
                if r['block_type']!='linear_attention':continue
                for h in range(32):
                    writer.writerow([r['benchmark'],r['method'],r['layer'],h,
                        *[r['means'].get(f'state_rank.{name}_per_head[{h}]') for name in ['r90','r95','effective_rank']]])
        for method in ['native','adapter']:
            for layer in range(32):
                cells=[lookup[(b,method,layer)] for b in benchmarks]
                lookup[('macro',method,layer)]={k:statistics.mean(c[k] for c in cells) for k in set.intersection(*(set(c) for c in cells))}
        linear=[i for i in range(32) if i%4!=3]
        fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
        for mi,method in enumerate(['native','adapter']):
            for bi,b in enumerate(benchmarks):
                matrix=np.array([[lookup[(b,method,l)][f'state_rank.r95_per_head[{h}]'] for h in range(32)] for l in linear])
                ax=axes[mi,bi];im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=128,cmap='viridis')
                ax.set(title=f'{titles[b]} / {method}',xlabel='Value head (0–31)',ylabel='Linear layer (0-based)')
                ax.set_yticks(range(24),linear,fontsize=7);fig.colorbar(im,ax=ax,label='Mean per-example r95')
        save(fig,'state_r95_head_heatmaps')
        fig,axes=plt.subplots(1,4,figsize=(18,3.8),constrained_layout=True)
        for ax,b in zip(axes,benchmarks+['macro']):
            for method in ['native','adapter']:
                ax.plot(linear,[lookup[(b,method,l)]['state_rank.r95'] for l in linear],marker='.',label=method)
            ax.set(title=titles[b],xlabel='Linear layer',ylabel='Head-mean r95 (maximum128)');ax.legend()
        save(fig,'state_r95_layer_curves')
        fig,axes=plt.subplots(2,4,figsize=(18,7),constrained_layout=True)
        for col,b in enumerate(benchmarks+['macro']):
            for key,label in [('representation_pre_norm.cosine','Hidden cosine'),('representation_post_norm.cosine','Postnorm hidden cosine'),('adapter_state_matched_context.cosine','Matched-state cosine')]:
                axes[0,col].plot(linear,[lookup[(b,'native',l)][key] for l in linear],marker='.',label=label)
            for key,label in [('representation_pre_norm.normalized_mse','Hidden NMSE'),('adapter_state_matched_context.normalized_mse','Matched-state NMSE')]:
                axes[1,col].plot(linear,[lookup[(b,'native',l)][key] for l in linear],marker='.',label=label)
            axes[0,col].set(title=titles[b],ylabel='Cosine',xlabel='Linear layer');axes[0,col].legend(fontsize=8)
            axes[1,col].set(ylabel='Normalized MSE',xlabel='Linear layer',yscale='symlog');axes[1,col].legend(fontsize=8)
        save(fig,'representation_vs_state_similarity')
        fig,axes=plt.subplots(1,4,figsize=(18,4),constrained_layout=True)
        for ax,b in zip(axes,benchmarks+['macro']):
            for method,color in [('native','tab:blue'),('adapter','tab:orange')]:
                for kind,ls,marker in [('LA',linear,'o'),('FA',list(range(3,32,4)),'s')]:
                    ax.plot(ls,[lookup[(b,method,l)]['text_effect_rank.r95_fraction'] for l in ls],marker=marker,color=color,linestyle='-' if kind=='LA' else '--',label=f'{method} {kind}')
            ax.set(title=titles[b],xlabel='Layer',ylabel='r95 / min(Ntext,2560)');ax.legend(fontsize=8)
        save(fig,'text_effect_normalized_ranks')
        # Per-layer reconstruction curves, retaining both oracle and adapter errors.
        fig,axes=plt.subplots(1,4,figsize=(18,4),constrained_layout=True)
        for ax,b in zip(axes,benchmarks+['macro']):
            ranks=config['ranks']
            ax.plot(ranks,[statistics.mean(lookup[(b,'native',l)][f'svd_reconstruction.{r}.normalized_mse'] for l in linear) for r in ranks],marker='o',label='Teacher SVD oracle')
            ax.axhline(statistics.mean(lookup[(b,'native',l)]['adapter_state_matched_context.normalized_mse'] for l in linear),ls='--',label='Adapter state')
            ax.set(title=titles[b],xlabel='Per-head rank',ylabel='State NMSE');ax.legend(fontsize=8)
        save(fig,'state_reconstruction_rank')
    accuracy,complete=load(run,'accuracy',config)
    report.extend(['',f'Accuracy: {len(accuracy)}/{expected} samples; complete={complete}.'])
    if complete:
        flat=[];agreements=defaultdict(list)
        for row in accuracy:
            refs={r['method']:r for r in row['variants'] if r['rank'] is None and r['method']!='state_adapter'}
            for v in row['variants']:
                label=v['method']+('' if v['rank'] is None else f'_r{v["rank"]}')
                flat.append(dict(benchmark=row['benchmark'],variant=label,score=v['score'],kl=v['kl_to_native'],unfinished=not v['stopped_by_eos']))
                if v['rank']==128:agreements[(row['benchmark'],v['method'])].append(v['generated_token_ids']==refs[v['method']]['generated_token_ids'])
        summary=summarize_groups(flat,['benchmark','variant']);(dest/'accuracy_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
        lookup={(r['benchmark'],r['variant']):r['means'] for r in summary}
        variants=list(dict.fromkeys(r['variant'] for r in summary))
        report.extend(['','| Variant | SQA | RealWorldQA | MMStar | Macro accuracy |','|---|---:|---:|---:|---:|'])
        for v in variants:
            scores=[100*lookup[(b,v)]['score'] for b in benchmarks]
            report.append('| '+v+' | '+' | '.join(f'{x:.2f}' for x in scores+[statistics.mean(scores)])+' |')
        report.extend(['','## Full-rank numeric control',''])
        for (b,m),eq in agreements.items():
            diff=100*(lookup[(b,m+'_r128')]['score']-lookup[(b,m)]['score'])
            report.append(f'- {b}/{m}: rank128-minus-original accuracy {diff:+.3f} percentage points; generated-token agreement {sum(eq)}/{len(eq)}.')
        fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
        for col,b in enumerate(benchmarks):
            for method in ['native','adapter']:
                ranks=config['ranks']
                axes[0,col].plot(ranks,[100*lookup[(b,f'{method}_r{r}')]['score'] for r in ranks],marker='o',label=method)
                axes[0,col].axhline(100*lookup[(b,method)]['score'],ls='--',alpha=.5)
                axes[1,col].plot(ranks,[lookup[(b,f'{method}_r{r}')]['kl'] for r in ranks],marker='o',label=method)
            axes[0,col].axhline(100*lookup[(b,'state_adapter')]['score'],color='black',ls=':',label='State-only adapter')
            axes[1,col].axhline(lookup[(b,'state_adapter')]['kl'],color='black',ls=':')
            axes[0,col].set(title=titles[b],ylabel='Accuracy (%)',xlabel='Per-head rank');axes[0,col].legend(fontsize=8)
            axes[1,col].set(ylabel='KL(native || variant)',xlabel='Per-head rank');axes[1,col].set_yscale('symlog',linthresh=1e-5)
        save(fig,'rank_accuracy_and_kl')
    sensitive,complete=load(run,'sensitivity',config)
    report.extend(['',f'Sensitivity: {len(sensitive)}/{expected} samples; complete={complete}.'])
    if complete:
        flat=[dict(benchmark=r['benchmark'],**v) for r in sensitive for v in r['perturbations']]
        summary=summarize_groups(flat,['benchmark','method','layer','block_type','magnitude'])
        (dest/'sensitivity_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
        curves={}
        for b in benchmarks:
            for method in ['native','adapter']:
                for kind in ['linear_attention','full_attention']:
                    for mag in config['perturbation_magnitudes']:
                        cells=[r['means'] for r in summary if r['benchmark']==b and r['method']==method and r['block_type']==kind and r['magnitude']==mag]
                        curves[b,method,kind,mag]={k:statistics.mean(c[k] for c in cells) for k in cells[0]}
        for method in ['native','adapter']:
            for kind in ['linear_attention','full_attention']:
                for mag in config['perturbation_magnitudes']:
                    cells=[curves[b,method,kind,mag] for b in benchmarks]
                    curves['macro',method,kind,mag]={k:statistics.mean(c[k] for c in cells) for k in cells[0]}
        fig,axes=plt.subplots(2,4,figsize=(18,7),constrained_layout=True)
        for col,b in enumerate(benchmarks+['macro']):
            for method,color in [('native','tab:blue'),('adapter','tab:orange')]:
                for kind,ls in [('linear_attention','-'),('full_attention','--')]:
                    mags=config['perturbation_magnitudes'];cells=[curves[b,method,kind,mag] for mag in mags]
                    for row,key in [(0,'text_hidden_error.normalized_frobenius'),(1,'next_token_kl')]:
                        axes[row,col].plot([c['actual_magnitude'] for c in cells],[c[key] for c in cells],marker='.',ls=ls,color=color,label=f'{method} {"LA" if kind=="linear_attention" else "FA"}')
            axes[0,col].set(title=titles[b],ylabel='Text hidden relative error',xlabel='Actual visual perturbation');axes[0,col].legend(fontsize=8)
            axes[1,col].set(ylabel='Next-token KL',xlabel='Actual visual perturbation');axes[1,col].set_yscale('symlog',linthresh=1e-6)
        save(fig,'perturbation_sensitivity')
        fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
        for ax,b in zip(axes,benchmarks):
            for method in ['native','adapter']:
                cells=sorted([r for r in summary if r['benchmark']==b and r['method']==method and r['magnitude']==.1],key=lambda r:r['layer'])
                ax.plot([r['layer'] for r in cells],[r['means']['next_token_kl'] for r in cells],marker='.',label=method)
            ax.set(title=titles[b],xlabel='Perturbed layer',ylabel='KL at 10% perturbation');ax.set_yscale('symlog',linthresh=1e-6);ax.legend()
        save(fig,'perturbation_layer_curves')
    (dest/'RESULTS.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report[:8]))

if __name__=='__main__':main()
