import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def stats(rows):
    valid=[r for r in rows if r['C_visual'] is not None]
    c=np.array([r['C_visual'] for r in valid]);stable=[r['C_visual'] for r in valid if r['denominator_above_rounding']]
    norm=[r['core_norm_ratio'] for r in valid]
    return dict(n=len(rows),valid=len(valid),C_median=float(np.median(c)),C_q25=float(np.quantile(c,.25)),
        C_q75=float(np.quantile(c,.75)),C_below_001_fraction=float(np.mean(c<.01)),
        C_replay_denominator_median=float(np.median([r['C_visual_replay_denominator'] for r in valid])),
        C_numerically_stable_median=float(np.median(stable)) if stable else None,numerically_stable_count=len(stable),
        ratio_of_absolute_sums=sum(abs(r['visual_signed_projection']) for r in valid)/sum(abs(r['total_signed_projection']) for r in valid),
        core_norm_ratio_median=float(np.median(norm)),projected_norm_ratio_median=float(np.median([r['projected_norm_ratio'] for r in valid])),
        negative_visual_projection_fraction=float(np.mean([r['visual_signed_projection']<0 for r in valid])),
        visual_signed_projection_mean=float(np.mean([r['visual_signed_projection'] for r in valid])),
        total_signed_projection_mean=float(np.mean([r['total_signed_projection'] for r in valid])))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();run=args.run
    out=run/'reports';out.mkdir(exist_ok=True);config=json.loads((run/'config.json').read_text())
    groups=defaultdict(list);curves=defaultdict(list);seen=set();maximum=defaultdict(float);margin_correct=0
    answer_rows=[]
    for path in sorted((run/'analysis').glob('realworldqa.shard*.jsonl')):
        with path.open() as f:
            for line in f:
                r=json.loads(line);assert r['index'] not in seen;seen.add(r['index']);assert r['native_first_token_exact']
                correct=r['final_logit_margin']>0;margin_correct+=correct
                for layer in r['layers']:
                    i=layer['layer']
                    for name,val in layer['checks'].items():maximum[name]=max(maximum[name],val)
                    for v in layer['positions']:
                        if v['C_visual'] is not None:curves[i,v['distance_after_visual']].append(v['C_visual'])
                        if v['answer_position']:
                            row=dict(v,index=r['index'],layer=i,gold=r['gold'],negative=r['negative'],final_logit_margin=r['final_logit_margin'])
                            answer_rows.append(row);groups[i].append(row);groups['all'].append(row)
                            groups['correct' if correct else 'incorrect'].append(row)
    assert seen==set(config['selected_indices']) and len(seen)==438
    table=[dict(layer=i,**stats(groups[i])) for i in sorted(k for k in groups if isinstance(k,int))]
    assert len(table)==24
    with (out/'layer_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    with (out/'answer_position_projections.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(answer_rows[0]));w.writeheader();w.writerows(answer_rows)
    with (out/'position_C_curves.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['layer','postvisual_position','n','C_median','C_q25','C_q75'])
        for (i,d),values in sorted(curves.items()):w.writerow([i,d,len(values),*np.quantile(values,[.5,.25,.75])])
    summary=dict(samples=438,open_ended_excluded=327,native_top_option_correct=margin_correct,
        primary_scope='Last prompt token, predicting first answer token; pooled 438 samples x 24 LA layers.',
        direction='Correct option letter minus highest-native-logit incorrect option letter; fixed per question.',
        overall=stats(groups['all']),correct_option_subset=stats(groups['correct']),incorrect_option_subset=stats(groups['incorrect']),
        maximum_numerical_errors=dict(maximum),native_first_token_exact_all=True,layer_summary=table)
    near13=[r for r in groups['all'] if .10<=r['core_norm_ratio']<=.20]
    if near13:summary['core_norm_10_to_20_percent_subset']=stats(near13)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    xx=np.arange(24);med=np.array([r['C_median'] for r in table]);lo=np.array([r['C_q25'] for r in table]);hi=np.array([r['C_q75'] for r in table])
    axes[0].fill_between(xx,lo,hi,color='#276a9e',alpha=.18,label='C interquartile range across questions')
    axes[0].plot(xx,med,color='#276a9e',marker='o',ms=3,label='Median answer-direction ratio C')
    axes[0].plot(xx,[r['projected_norm_ratio_median'] for r in table],color='#c16a2e',label='Median output norm ratio')
    axes[0].set_xticks(xx,[r['layer'] for r in table],rotation=90)
    axes[0].set(xlabel='Linear-attention layer (zero-based)',ylabel='Ratio',title='Answer position: local projection versus norm');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    valid=[r for r in answer_rows if r['C_visual'] is not None and r['C_visual']>0]
    im=axes[1].hexbin([r['core_norm_ratio'] for r in valid],[r['C_visual'] for r in valid],yscale='log',gridsize=45,mincnt=1,cmap='viridis',bins='log')
    axes[1].axhline(.01,color='#b23b3b',ls='--',lw=1,label='C = 0.01');axes[1].legend(fontsize=8)
    axes[1].set(xlabel='Core visual-source norm ratio',ylabel='Answer-direction ratio C (log scale)',title='Each point: one question × one layer');fig.colorbar(im,ax=axes[1],label='Count (log color scale)')
    fig.suptitle('Qwen3.5-4B · RealWorldQA 438 multiple-choice questions · DeepStack off\nLocal unembedding projection; not a downstream causal effect',fontsize=11)
    for suffix in ['png','pdf']:fig.savefig(out/f'answer_projection.{suffix}',dpi=190)
    plt.close(fig)
    text=['# Visual-source answer-direction projection','',
        '438 multiple-choice questions from the fixed RealWorldQA 765-item manifest; 327 open-ended questions are not included because no wrong-answer candidate was specified. Native Qwen3.5-4B, DeepStack off. No model intervention.',
        '', 'The 4096-dimensional core readout is mapped into 2560 residual channels using the same native total-readout RMS denominator, gate, norm weight and output projection for visual and text sources. d is the correct option unembedding row minus the highest-native-logit incorrect option row.',
        '', '**This local projection does not include downstream layers or final RMSNorm and cannot by itself establish causal importance to the final decision.**',
        '', '| Layer | Median C | C < 0.01 | Median core norm ratio | Median mapped norm ratio |','|---|---:|---:|---:|---:|']
    for r in table:text.append(f"| {r['layer']} | {r['C_median']:.4f} | {r['C_below_001_fraction']:.2%} | {r['core_norm_ratio_median']:.4f} | {r['projected_norm_ratio_median']:.4f} |")
    text += ['', 'Primary statistics use the final prompt position that predicts the first answer token. Signed numerators and denominators are retained in answer_position_projections.csv. Near-zero denominators are flagged; robustness against native-vs-replay numerical differences is reported separately. Numeric C values are not clipped.',
             '', '![Answer projection](answer_projection.png)']
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='layer_summary'},indent=2))


if __name__=='__main__':main()
