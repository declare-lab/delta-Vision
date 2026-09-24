"""Paired inference statistics and native token/head diagnostic artifacts."""
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
from scipy.stats import binomtest

LA=[i for i in range(32) if i%4!=3]
METRICS=['beta','retention','value_norm','residual_norm','delta_write_norm','previous_state_norm',
         'relative_delta_write','native_minus_skip_state_norm','native_readout_norm',
         'native_minus_skip_readout_norm','relative_readout','query_state_alignment']


def read(p):
    return [json.loads(l) for l in p.read_text().splitlines(keepends=True) if l.endswith('\n')]


def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def dump(p,v):p.write_text(json.dumps(clean(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def mean(x,axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);return np.nanmean(x,axis=axis)


def paired(native,other):
    a=np.asarray(native);b=np.asarray(other);d=b-a
    assert set(a)<=set([0,1]) and set(b)<=set([0,1])
    up=int(((a==0)&(b==1)).sum());down=int(((a==1)&(b==0)).sum())
    vals,n=np.unique(d,return_counts=True)
    boot=np.random.default_rng(44).multinomial(len(d),n/len(d),size=10000)@vals/len(d)*100
    return dict(native_accuracy=100*float(a.mean()),accuracy=100*float(b.mean()),delta_pp=100*float(d.mean()),
        improved=up,worsened=down,both_correct=int(((a==1)&(b==1)).sum()),both_wrong=int(((a==0)&(b==0)).sum()),
        mcnemar_exact_p=float(binomtest(up,up+down,.5).pvalue) if up+down else 1.,
        paired_bootstrap_95ci_pp=np.quantile(boot,[.025,.975]).tolist())


def analyze_traces(run,dest,records,accuracy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cache=dest/'phase_head_sample_means.npz'
    phases=[0,1,2,3,4,6,5];sizes=[16,1,64,1,32,8,8]
    if not cache.exists():
        sample=np.full((765,24,7,32,12),np.nan,dtype=np.float32)
        acc=np.zeros((24,sum(sizes),32,12));count=np.zeros_like(acc)
        for ordinal,r in enumerate(records):
            with np.load(r['trace']) as z:m=z['metrics'];p=z['phase']
            begin=0
            for phase,n in zip(phases,sizes):
                selected=m[:,p==phase]
                if selected.shape[1]:sample[r['index'],:,phase]=mean(selected,axis=1)
                for j,ids in enumerate(np.array_split(np.arange(selected.shape[1]),n)):
                    if not len(ids):continue
                    v=mean(selected[:,ids],axis=1);valid=np.isfinite(v)
                    acc[:,begin+j]+=np.where(valid,v,0);count[:,begin+j]+=valid
                begin+=n
            if ordinal%100==0:print('aggregate traces',ordinal,flush=True)
        aligned=np.divide(acc,count,out=np.full_like(acc,np.nan),where=count>0)
        np.savez_compressed(cache,sample=sample,aligned=aligned,aligned_counts=count,phases=phases,phase_bins=sizes)
    with np.load(cache) as z:sample=z['sample'];aligned=z['aligned']
    # Sample is the inferential unit; heads/positions are not treated as independent examples.
    layer=mean(sample,axis=(0,3))
    dump(dest/'native_layer_phase_metrics.json',[
        dict(layer=i,phases={str(p):dict(zip(METRICS,layer[j,p].tolist())) for p in range(7)}) for j,i in enumerate(LA)])
    native=accuracy['native'];cohorts={}
    for method in ['boundary_rank0','state_skip']:
        if method not in accuracy:continue
        a=np.array(native);b=np.array(accuracy[method]);groups={
            'both_correct':(a==1)&(b==1),'native_only':(a==1)&(b==0),
            'improved':(a==0)&(b==1),'both_wrong':(a==0)&(b==0)}
        cohorts[method]={name:dict(samples=int(mask.sum()),indices=np.flatnonzero(mask).tolist()) for name,mask in groups.items()}
        arrays={name:mean(sample[mask],axis=0) for name,mask in groups.items() if mask.any()}
        np.savez_compressed(dest/f'{method}_paired_cohort_head_metrics.npz',**arrays)
    dump(dest/'paired_cohorts.json',cohorts)
    marker=dest/'trace_figures.done'
    if not marker.exists():
        fig,axs=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
        for ax,k in zip(axs.flat,[0,1,4,6,10,11]):
            for p,label in [(2,'Image patches'),(4,'Question text'),(5,'Answer inputs')]:
                ax.plot(LA,layer[:,p,k],marker='o',markersize=2,label=label)
            if k in [4,6,10]:ax.set_yscale('log')
            ax.set(xlabel='GDN layer (0-based)',ylabel=METRICS[k]);ax.grid(alpha=.2)
        axs[0,0].legend(fontsize=8)
        for ext in ['png','pdf']:fig.savefig(dest/f'layer_write_forget_read.{ext}',dpi=170)
        plt.close(fig)
        # Every head retained. Across-example x-axis uses phase-aligned bins.
        cuts=np.cumsum([0]+sizes);names=['Prefix','Start','Image','End','Question','Template','Answer']
        for j,i in enumerate(LA):
            fig,axs=plt.subplots(3,2,figsize=(14,9),constrained_layout=True)
            for ax,k in zip(axs.flat,[0,1,4,6,10,11]):
                values=aligned[j,:,:,k].T
                if k in [4,6,10]:values=np.log10(np.maximum(values,1e-12))
                im=ax.imshow(values,aspect='auto',interpolation='nearest')
                ax.set(ylabel='Head',title=('log10 ' if k in [4,6,10] else '')+METRICS[k])
                ax.set_xticks((cuts[:-1]+cuts[1:]-1)/2,names,rotation=20)
                for x in cuts[1:-1]:ax.axvline(x-.5,color='white',linewidth=.6)
                fig.colorbar(im,ax=ax,shrink=.8)
            fig.suptitle(f'RealWorldQA765: layer {i}; phase-aligned position bins, all32 heads')
            fig.savefig(dest/f'layer{i:02d}_all_heads.png',dpi=140);plt.close(fig)
        # Actual sequence positions for a pinned example, not normalized bins.
        r=next(r for r in records if r['index']==0)
        with np.load(r['trace']) as z:m=z['metrics']
        fig,axs=plt.subplots(3,1,figsize=(13,8),constrained_layout=True)
        for ax,k in zip(axs,[0,1,10]):
            v=mean(m[:,:,:,k],axis=2)
            if k==10:v=np.log10(np.maximum(v,1e-12))
            im=ax.imshow(v,aspect='auto');ax.set_yticks(range(24),LA)
            ax.set(xlabel='Actual sequence position',ylabel='GDN layer',title=METRICS[k]);fig.colorbar(im,ax=ax)
            ax.axvline(r['visual_start']-.5,color='white');ax.axvline(r['visual_end']-.5,color='white')
        fig.savefig(dest/'example0000_actual_positions.png',dpi=180);plt.close(fig);marker.write_text('complete\n')


def analyze_spectra(run,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=dest/'spectrum_summary.json'
    if target.exists():return
    ranks=[];ers=[];cos=[];relative=[];energy=[]
    for p in sorted((run/'spectra').glob('*.npz')):
        with np.load(p) as z:
            ranks.append(z['r95']);ers.append(z['effective_rank']);cos.append(z['pre_post_cosine']);relative.append(z['pre_post_relative_change'])
            s=z['singular_values'];ss=s*s;energy.append(np.divide(ss,ss.sum(-1,keepdims=True),out=np.zeros_like(ss),where=ss.sum(-1,keepdims=True)>0))
    assert len(ranks)==765
    r=np.mean(ranks,axis=(0,3));e=np.mean(ers,axis=(0,3));en=np.mean(energy,axis=(0,3))
    co=mean(np.array(cos),axis=(0,2));re=mean(np.array(relative),axis=(0,2))
    dump(target,dict(samples=765,full_rank=128,layers=[dict(layer=i,r95=r[j].tolist(),effective_rank=e[j].tolist(),
        pre_post_cosine=float(co[j]),pre_post_relative_change=float(re[j])) for j,i in enumerate(LA)],
        kinds=['before_vision_start','after_vision_end','skip_after_vision_end','native_minus_skip_effect']))
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for k,name in enumerate(['Pre','Post','Skip post','Visual effect']):
        axs[0].plot(LA,r[:,k],label=name);axs[1].plot(LA,e[:,k],label=name)
    for j,i in enumerate(LA):axs[2].plot(range(1,129),en[j,3].cumsum(),alpha=.5,label=str(i))
    axs[0].set(xlabel='Layer',ylabel='r95 (full128)');axs[1].set(xlabel='Layer',ylabel='Effective rank (full128)')
    axs[2].set(xlabel='Rank',ylabel='Visual-effect cumulative squared energy');axs[0].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.2)
    for ext in ['pdf','png']:fig.savefig(dest/f'state_spectra.{ext}',dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--partial',action='store_true');a=p.parse_args();run=a.run
    c=json.loads((run/'config.json').read_text());dest=run/'reports';dest.mkdir(exist_ok=True)
    grouped={};finished=[]
    for stage in c['stage_order']:
        if not (run/'audits'/f'{stage}.json').exists():continue
        finished.append(stage)
        if stage in ['analysis','spectrum']:continue
        records=[r for f in (run/stage).glob('shard*.jsonl') for r in read(f)]
        assert len(records)==765
        for r in records:
            for name,v in r['variants'].items():
                group=grouped.setdefault(name,{})
                if r['index'] in group:assert group[r['index']]['generated_token_ids']==v['generated_token_ids']
                group[r['index']]=v
    if not a.partial:assert finished==c['stage_order']
    if 'native' not in grouped:return
    accuracy={name:[g[i]['score'] for i in range(765)] for name,g in grouped.items()}
    summary={name:paired(accuracy['native'],acc) for name,acc in accuracy.items()}
    for name,s in summary.items():
        s['invalid']=sum(v['invalid'] for v in grouped[name].values())
        s['mean_kl']=float(np.mean([v.get('kl_to_native',0.) for v in grouped[name].values()]))
    # Holm multiplicity adjustment for the accuracy intervention comparisons.
    ordered=sorted([n for n in summary if n!='native'],key=lambda n:summary[n]['mcnemar_exact_p']);prev=0.
    for i,name in enumerate(ordered):
        prev=max(prev,min(1.,(len(ordered)-i)*summary[name]['mcnemar_exact_p']));summary[name]['mcnemar_holm_p']=prev
    result=dict(complete=not a.partial,completed_stages=finished,samples=765,results=summary,group_aliases=c['group_aliases'])
    names=['native','state_skip','fa_off','state_skip_fa_off']
    if all(n in accuracy for n in names):
        d=np.array(accuracy[names[3]])-accuracy[names[2]]-np.array(accuracy[names[1]])+accuracy[names[0]]
        vals,ct=np.unique(d,return_counts=True)
        boot=np.random.default_rng(44).multinomial(765,ct/765,size=10000)@vals/765*100
        result['factorial_interaction']=dict(effect_pp=float(d.mean()*100),paired_bootstrap_95ci_pp=np.quantile(boot,[.025,.975]).tolist(),
            definition='Accuracy(skip,FAoff)-Accuracy(native,FAoff)-Accuracy(skip,FAon)+Accuracy(native,FAon)')
    dump(dest/'RESULTS.json',result)
    lines=['# Qwen3.5-4B RealWorldQA write / forget / read','',
        '765 fixed examples; DeepStack off; FA2/FLA; native model only. Original cap8, greedy, unfinished/no-EOS zero. Paired comparisons on identical samples.',
        'State-skip edits visual-position g=beta=0 throughout each selected layer. Historical boundary-rank0 restores pre-image memory only after image processing and preserves visual readouts. These are distinct interventions.',
        'Full-attention OFF masks text-query access to visual KV only; visual queries, caches, FFNs remain. Conv-after-end and conv-before-end are separate controls; the latter prevents image patches from leaking through the end marker via this local convolution.',
        '', '| Condition | Acc (%) | Δ pp | Improved | Worsened | Exact McNemar p | Paired95% CI Δpp | Invalid |',
        '|---|---:|---:|---:|---:|---:|---|---:|']
    for name,s in summary.items():
        ci=s['paired_bootstrap_95ci_pp']
        lines.append(f'| {name} | {s["accuracy"]:.2f} | {s["delta_pp"]:+.2f} | {s["improved"]} | {s["worsened"]} | {s["mcnemar_exact_p"]:.4g} | [{ci[0]:+.2f}, {ci[1]:+.2f}] | {s["invalid"]} |')
    lines+=['','Group aliases (duplicate interventions run once):',json.dumps(c['group_aliases'],ensure_ascii=False),
        '', 'Raw p-values shown above; Holm-adjusted values are in RESULTS.json. No head/token is treated as an independent example. Layer/head cohort contrasts are descriptive, not evidence of significant localization by themselves.']
    if 'factorial_interaction' in result:lines+=['',f'Factorial interaction: {result["factorial_interaction"]}']
    (dest/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if 'analysis' in finished:
        records=[r for f in (run/'analysis').glob('shard*.jsonl') for r in read(f)]
        analyze_traces(run,dest,records,accuracy)
    if 'spectrum' in finished:analyze_spectra(run,dest)
    print(dest/'RESULTS.md',flush=True)


if __name__=='__main__':main()
