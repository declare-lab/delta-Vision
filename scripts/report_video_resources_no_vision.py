"""Report the requested FLOP scope without disturbing active GPU workers.

Subtract the native vision cost of the identical input from every method.
This keeps method-specific selection work, even if implemented inside vision.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import statistics
import time


def atomic_text(path, text):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(text)
    temp.replace(path)


def load_rows(paths):
    rows = [json.loads(line) for path in sorted(paths) for line in path.read_text().splitlines()]
    result = {r['index']: r for r in rows}
    assert len(result) == len(rows), 'Duplicate indices'
    return result


def render(root, status):
    cases = status.get('completed_cases', [])
    if not cases:
        return
    names = dict(base='Qwen3-VL-4B', adapter='Embedding Adapter', fastv='FastV', dart='DART',
        visionzip='VisionZip', divprune='DivPrune', zoo='ZOO-Prune', sparsevlm='SparseVLM')
    base = load_rows((root/'base').glob('flops_*.jsonl'))
    assert len(base) == 999
    records = []
    for case in cases:
        method, retention = case['method'], case['retention']
        name = method if method in ('base', 'adapter') else f'{method}_r{round(100*retention):03d}'
        folder = root/name
        flops = load_rows(folder.glob('flops_*.jsonl'))
        memory = load_rows(folder.glob('optimized_*.jsonl' if method in ('base','adapter') else f'{name}/fa2_metadata_*.jsonl'))
        assert set(flops) == set(memory) == set(base)
        for i in base:
            assert flops[i]['input_sha256'] == memory[i]['input_sha256'] == base[i]['input_sha256']
            assert flops[i]['tokens'] == memory[i]['tokens']
        prefill = statistics.mean(flops[i]['prefill_matrix_flops']-base[i]['vision_matrix_flops'] for i in base)/1e12
        decode = statistics.mean(r['decode_matrix_flops'] for r in flops.values())/1e12
        records.append(dict(method=names[method], retention='adapter' if method=='adapter' else f'{retention:.0%}',
            samples=len(flops), peak_allocated_GiB=max(t['peak_memory_mb'] for r in memory.values() for t in r['trials'])/1024,
            old_decoder_prefill_TFLOPs=statistics.mean(r['old_decoder_prefill_flops'] for r in flops.values())/1e12,
            prefill_no_vision_TFLOPs=prefill, decode_TFLOPs=decode, total_no_vision_TFLOPs=prefill+decode))
    for r in records:
        r['prefill_no_vision_pct_base']=100*r['prefill_no_vision_TFLOPs']/records[0]['prefill_no_vision_TFLOPs']
        r['total_no_vision_pct_base']=100*r['total_no_vision_TFLOPs']/records[0]['total_no_vision_TFLOPs']
        r['old_decoder_prefill_pct_base']=100*r['old_decoder_prefill_TFLOPs']/records[0]['old_decoder_prefill_TFLOPs']
    atomic_text(root/'resource_summary_no_vision.json',json.dumps(records,indent=2)+'\n')
    csv_buffer=io.StringIO();writer=csv.DictWriter(csv_buffer,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    atomic_text(root/'resource_summary_no_vision.csv',csv_buffer.getvalue())
    lines=['# Video-MME：FLOPs 统一排除视觉编码', '', f'已完成并校验：{len(records)}/14 组，每组 999 条。', '',
        '所有方法（包括 base）统一扣除同一输入下的原生视觉编码 FLOPs。保留语言模型、adapter、筛选/合并、输出层及 decode 的矩阵计算；VisionZip 在视觉模块内额外计算的筛选信息仍计入方法开销。',
        '2 FLOPs/MAC；attention 使用 dense QK/AV 口径；不计标量 norm/softmax/激活及排序、索引操作。旧 decoder prefill 公式单列为历史参考。',
        '本次只修改 FLOPs 报告范围。Peak Mem 仍为实际完整推理峰值，包含模型权重和 graph 缓存，未做显存扣减。', '',
        '| 方法 | 保留率 | Peak Mem (GiB) | Prefill FLOPs（不含视觉，T） | Decode FLOPs（T） | 合计 FLOPs（不含视觉，T） | 合计 / Base |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['method']} | {r['retention']} | {r['peak_allocated_GiB']:.3f} | {r['prefill_no_vision_TFLOPs']:.4f} | {r['decode_TFLOPs']:.4f} | {r['total_no_vision_TFLOPs']:.4f} | {r['total_no_vision_pct_base']:.2f}% |")
    atomic_text(root/'RESULTS_NO_VISION.md','\n'.join(lines)+'\n')
    return records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    previous=None
    while True:
        status=json.loads((args.output/'status.json').read_text())
        key=json.dumps(status.get('completed_cases',[]),sort_keys=True)
        if key!=previous:
            records=render(args.output,status)
            print(json.dumps(dict(completed_cases=len(records or []),report='RESULTS_NO_VISION.md')),flush=True)
            previous=key
        if not args.watch or status['state'] in ('complete','failed','failed_validation','stopped_superseded'):
            break
        time.sleep(5)


if __name__=='__main__':main()
