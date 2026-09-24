"""Report completed full-pipeline records with Total = Prefill + Decode.

Keep both FLOP scopes explicit; timing includes fresh visual encoding.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

CASES={'base':'Base','adapter':'完整 adapter','first5_last10':'关闭前5＋后10层视觉注入','first10_last10':'关闭前10＋后10层视觉注入'}

def report(source, output):
    output.mkdir(parents=True,exist_ok=True)
    result=[];hashes={}
    def read(case,pattern):
        rows={}
        for p in sorted((source/case).glob(pattern)):
            data=p.read_bytes();hashes[str(p)]=hashlib.sha256(data).hexdigest()
            for line in data.splitlines():
                row=json.loads(line);assert row['index'] not in rows;rows[row['index']]=row
        assert set(rows)==set(range(999)),(case,pattern,len(rows))
        return rows
    base=None
    for case,label in CASES.items():
        memory=read(case,'optimized_*.jsonl');flops=read(case,'flops_*.jsonl')
        if base is None:base=flops
        for i in memory:
            assert memory[i]['input_sha256']==flops[i]['input_sha256']==base[i]['input_sha256']
            assert memory[i]['tokens']==flops[i]['tokens']
            assert memory[i]['timed_captures']==memory[i]['timed_fallbacks']==0
            assert len(memory[i]['trials'])==3 and len(memory[i]['tokens'])==8
            assert flops[i]['vision_matrix_flops']==base[i]['vision_matrix_flops']
        keys=['request_prefill_time_s','decode_time_s'] if case=='base' else ['prefill_s','decode_s']
        row=dict(case=case,method=label,samples=999)
        for name,key in zip(['prefill_ms','decode_ms'],keys):
            row[name]=1000*statistics.mean(statistics.median(t[key] for t in x['trials']) for x in memory.values())
        row['total_ms']=row['prefill_ms']+row['decode_ms']
        row['peak_GiB']=max(t['peak_memory_mb'] for x in memory.values() for t in x['trials'])/1024
        row['flops_with_vision_T']=statistics.mean(x['request_matrix_flops'] for x in flops.values())/1e12
        row['flops_without_vision_T']=statistics.mean(x['request_matrix_flops']-x['vision_matrix_flops'] for x in flops.values())/1e12
        result.append(row)
    for row in result:
        for key in ['prefill','decode','total']:row[key+'_speedup']=result[0][key+'_ms']/row[key+'_ms']
        for scope in ['with','without']:row['flops_'+scope+'_vision_pct_base']=100*row['flops_'+scope+'_vision_T']/result[0]['flops_'+scope+'_vision_T']
    (output/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 包含视觉编码：Video-MME 999 条','',
        '最新时间口径：Prefill 包含视觉编码；Total = Prefill + Decode，不使用整请求墙钟时间。每题三次测量，各阶段取中位数后对999题求平均。Decode 为7次 forward，共生成8 token。',
        'FA2、DeepStack关闭、adapter fast-path、CUDA Graph；同一8卡逐组测量。Peak为完整流程最大 allocated GiB，含驻留权重及graph缓存。FLOPs的含视觉编码与不含视觉编码两种口径分别列出，避免混淆。','',
        '| 方法 | Prefill ms / 加速 | Decode ms / 加速 | Total ms / 加速 | Peak GiB | FLOPs含视觉 T | FLOPs不含视觉 T |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in result:
        times=[f"{r[k+'_ms']:.2f} / {r[k+'_speedup']:.3f}×" for k in ['prefill','decode','total']]
        lines.append(f"| {r['method']} | {' | '.join(times)} | {r['peak_GiB']:.3f} | {r['flops_with_vision_T']:.4f} | {r['flops_without_vision_T']:.4f} |")
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (output/'AUDIT.json').write_text(json.dumps(dict(passed=True,samples_per_case=999,cases=list(CASES),
        timing_includes_vision=True,total_definition='mean of per-input prefill median plus decode median',
        input_and_generated_token_checks=True,timed_captures=0,timed_fallbacks=0,source_run=str(source),
        raw_file_sha256=hashes),indent=2)+'\n')
    print((output/'RESULTS.md').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.source.resolve(),a.output.resolve())
