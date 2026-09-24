"""Corrected scope: prepared visual embeddings -> LLM prefill + cached decode.

Encoder computation and its graph pools are outside the measured region.
Total is the sum of stage medians, not end-to-end request wall time.
"""
import argparse,json,os,statistics
from pathlib import Path
ROOT=Path(os.environ.get('RESOURCE_REPO',str(Path(__file__).resolve().parents[3])))
CASES=['base','adapter','first5_last10','first10_last10']

def dump(p,v):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)

def read_rows(paths):
    out={}
    for p in paths:
        for s in p.read_text().splitlines():
            x=json.loads(s);assert x['index'] not in out;out[x['index']]=x
    return out

def report(a):
    result=[]
    for case in CASES:
        mem=read_rows((a.run/case).glob('shard*.jsonl'));flop=read_rows((a.previous/case).glob('flops_*.jsonl'))
        assert set(mem)==set(flop)==set(range(999))
        for i in mem:
            assert mem[i]['input_sha256']==flop[i]['input_sha256'] and mem[i]['tokens']==flop[i]['tokens']
            assert mem[i]['timed_encoder_calls']==mem[i]['timed_captures']==mem[i]['timed_fallbacks']==0
        row=dict(case=case,samples=999,flops_T=statistics.mean(x['request_matrix_flops']-x['vision_matrix_flops'] for x in flop.values())/1e12,
            peak_GiB=max(t['peak_GiB'] for x in mem.values() for t in x['trials']))
        for k in ['prefill','decode']:row[k+'_ms']=1000*statistics.mean(statistics.median(t[k+'_s'] for t in x['trials']) for x in mem.values())
        row['total_ms']=row['prefill_ms']+row['decode_ms'];result.append(row)
    for r in result:
        r['flops_pct_base']=100*r['flops_T']/result[0]['flops_T']
        for k in ['prefill','decode','total']:r[k+'_speedup']=result[0][k+'_ms']/r[k+'_ms']
    dump(a.run/'RESULTS.json',result)
    lines=['# Corrected LLM-only Video-MME 999 resources','','Visual encoding and multimodal position preparation are outside timing. Total = LLM prefill + seven decode forwards. Stage times: median of three trials per input, then mean over 999 inputs. Same FA2/DeepStack-off/fast-path/CUDA graph setup. Peak = allocated memory during LLM prefill/decode, including resident model weights and LLM graph pools; no vision graph retained. FLOPs exclude visual encoding.','',
        '| Method | FLOPs (T) | FLOPs/base | Peak (GiB) | Prefill (ms) | Decode (ms) | Total (ms) | Prefill speedup | Decode speedup | Total speedup |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in result:lines.append(f"| {r['case']} | {r['flops_T']:.4f} | {r['flops_pct_base']:.2f}% | {r['peak_GiB']:.3f} | {r['prefill_ms']:.2f} | {r['decode_ms']:.2f} | {r['total_ms']:.2f} | {r['prefill_speedup']:.3f} | {r['decode_speedup']:.3f} | {r['total_speedup']:.3f} |")
    (a.run/'RESULTS.md').write_text('\n'.join(lines)+'\n')



def main():
    p=argparse.ArgumentParser(description='Join layer-ablation LLM timing with matched counted FLOPs. Run timing via the llm profile.')
    p.add_argument('--run',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    report(p.parse_args())

if __name__=='__main__':main()
