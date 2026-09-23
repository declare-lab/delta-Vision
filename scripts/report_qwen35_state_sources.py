"""Two source-attribution curves, layer heatmaps, and auditable numeric tables."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run
    out=run/'reports';out.mkdir(exist_ok=True)
    curves=defaultdict(lambda:np.zeros(5,dtype=np.float64))
    sample_means=defaultdict(list);max_checks=defaultdict(float);seen=set();correct=0;over_one=total_positions=0
    for f in sorted((run/'analysis').glob('realworldqa.shard*.jsonl')):
        with f.open() as handle:
            for line in handle:
                r=json.loads(line);assert r['index'] not in seen;seen.add(r['index'])
                assert r['native_generation_exact'];correct+=r['generated']['score']
                for layer in r['layers']:
                    i=layer['layer'];records=[v for v in layer['positions'] if v['distance_after_visual']>0]
                    assert all(records[j]['visual_state_survival']<=records[j-1]['visual_state_survival']+2e-5 for j in range(1,len(records)))
                    for check in layer['checks']:
                        for name,val in check.items():max_checks[name]=max(max_checks[name],val)
                    for phase in ['prompt','decode']:
                        positions=[v for v in records if v['phase']==phase]
                        if positions:
                            sample_means[i,phase+'_readout'].append(np.mean([v['visual_readout_ratio'] for v in positions]))
                            sample_means[i,phase+'_native_readout'].append(np.mean([v['visual_readout_ratio_native_denominator'] for v in positions]))
                    for v in records:
                        d=v['distance_after_visual'];vr=v['visual_readout_ratio'];sr=v['visual_state_survival']
                        curves[i,d,'all']+=np.array([1,vr,sr,v['visual_readout_ratio_native_denominator'],vr>1])
                        if v['phase']=='decode':
                            decode_step=v['position']-r['prompt_length']+1
                            curves[i,decode_step,'decode']+=np.array([1,vr,sr,v['visual_readout_ratio_native_denominator'],vr>1])
                        over_one+=vr>1;total_positions+=1
                        if d in [1,8,16,32]:sample_means[i,f'state_at_{d}'].append(sr)
    assert seen==set(range(765))
    layers=sorted({key[0] for key in curves});assert len(layers)==24
    table=[]
    for i in layers:
        row={'layer':i}
        for name in ['prompt_readout','decode_readout','prompt_native_readout','decode_native_readout','state_at_1','state_at_8','state_at_16','state_at_32']:
            vals=sample_means[i,name];row[name]=float(np.mean(vals)) if vals else None;row[name+'_n']=len(vals)
        table.append(row)
    with (out/'layer_summary.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    with (out/'position_curves.csv').open('w') as h:
        w=csv.writer(h);w.writerow(['layer','scope','position','samples','visual_readout_ratio','visual_state_survival','native_denominator_ratio','fraction_ratio_above_one'])
        for (i,d,phase),a in sorted(curves.items()):w.writerow([i,phase,d,int(a[0]),*(a[1:]/a[0])])
    aggregate={name:float(np.mean([r[name] for r in table if r[name] is not None])) for name in ['prompt_readout','decode_readout','prompt_native_readout','decode_native_readout','state_at_1','state_at_8','state_at_16','state_at_32']}
    summary=dict(samples=765,layers=24,native_accuracy=100*correct/765,native_generation_exact_all=True,
        aggregate=aggregate,maximum_numerical_errors=dict(max_checks),
        readout_ratio_above_one_fraction=over_one/total_positions,
        norm_scope='Readout L2 across 32x128 head dimensions; state Frobenius across 32x128x128; before gated RMSNorm/out_proj.',
        attribution='By position of additive write B, using fixed native q/k/v/g/beta and complete A for both sources. Not semantic origin: text-position writes may already encode image information.',
        averaging='Layer summaries: mean positions within each sample, then mean samples. Overall summary equally averages the 24 layer means. Position curves condition on samples reaching that position.',
        layer_summary=table)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    for phase,stem,xlabel in [('all','source_curves','Text position after visual segment'),('decode','decode_source_curves','Cached decode input step')]:
        xmax=max(d for _,d,s in curves if s==phase)
        values=np.full((24,xmax,2),np.nan);counts=np.zeros(xmax)
        for li,i in enumerate(layers):
            for d in range(1,xmax+1):
                a=curves.get((i,d,phase))
                if a is not None:values[li,d-1]=a[1:3]/a[0];counts[d-1]=a[0]
        xs=np.arange(1,xmax+1);fig,axes=plt.subplots(1,2,figsize=(12,4.3),constrained_layout=True)
        for j,(ax,color,label) in enumerate(zip(axes,['#2469a0','#a04721'],
                [r'$\|o^{vis}_t\|_2/\|o_t\|_2$',r'$\|S^{vis}_t\|_F/\|S^{vis}_{end-vis}\|_F$'])):
            avg=np.nanmean(values[:,:,j],axis=0);lo=np.nanpercentile(values[:,:,j],10,axis=0);hi=np.nanpercentile(values[:,:,j],90,axis=0)
            ax.fill_between(xs,np.maximum(lo,1e-12) if j else lo,np.maximum(hi,1e-12) if j else hi,color=color,alpha=.15,label='10–90% across layer means')
            ax.plot(xs,np.maximum(avg,1e-12) if j else avg,color=color,lw=2,label='Mean of 24 layer curves')
            ax.set(xlabel=xlabel,ylabel=label,title=['Visual-source readout','Visual-source state survival'][j]);ax.grid(alpha=.2)
            if j:ax.set_yscale('log')
            ax.legend(fontsize=8)
        fig.suptitle('Qwen3.5-4B · RealWorldQA (765) · DeepStack off\nFixed vanilla trajectory; norms pooled across heads',fontsize=12)
        for suffix in ['png','pdf']:fig.savefig(out/f'{stem}.{suffix}',dpi=190)
        plt.close(fig)
        if phase=='all':
            fig,axes=plt.subplots(2,1,figsize=(12,9),constrained_layout=True)
            for j,ax in enumerate(axes):
                val=values[:,:,j] if j==0 else np.log10(np.maximum(values[:,:,j],1e-12))
                im=ax.imshow(val,aspect='auto',origin='upper',extent=[.5,xmax+.5,23.5,-.5],cmap='viridis' if j==0 else 'magma')
                ax.set_yticks(range(24),layers);ax.set(xlabel=xlabel,ylabel='Linear-attention layer',title=['Visual-source readout ratio','log10 visual-state survival (display floor 1e-12)'][j]);fig.colorbar(im,ax=ax)
            for suffix in ['png','pdf']:fig.savefig(out/f'layer_heatmaps.{suffix}',dpi=190)
            plt.close(fig)
            with (out/'position_sample_counts.csv').open('w') as h:
                w=csv.writer(h);w.writerow(['position_after_visual','samples']);w.writerows(zip(xs,counts.astype(int)))
    text=['# Gated DeltaNet write-source decomposition — RealWorldQA','',
          'Native Qwen3.5-4B, DeepStack disabled; 765 examples; all 24 linear-attention layers and 32 heads. Native generated token IDs unchanged on every example.',
          '', 'A_t = exp(g_t) (I - beta_t k_t k_t^T); B_t = beta_t k_t v_t^T. Both source states receive the full same A_t, including key-dependent erasure. Only B_t is routed by its visual/text token position.',
          '', '**This is write-position attribution. Text-position writes can already carry image information through convolution or prior layers.**',
          '', '| Quantity | Mean |','|---|---:|']
    text += [f'| {name} | {val:.6g} |' for name,val in aggregate.items()]
    text += ['', 'The two source readouts need not be orthogonal: the readout norm ratio can exceed 1. No clipping is applied to numeric data. State survival normalizes by the end-of-image visual-source state, and the plotted log scale alone has a 1e-12 display floor.',
             '', 'The direct FP32 recurrence is algebraically additive. Differences from native BF16 chunk execution are reported separately in summary.json; the model always receives its unmodified native output/cache.',
             '', 'Position curves average only examples that reach a position; sample counts are provided in position_sample_counts.csv. Readouts are before gated RMSNorm and output projection.',
             '', '![Two source curves](source_curves.png)','', '![Layer heatmaps](layer_heatmaps.png)']
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='layer_summary'},indent=2))


if __name__=='__main__':main()
