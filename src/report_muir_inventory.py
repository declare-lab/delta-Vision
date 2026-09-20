"""Read existing scores only; never launch evaluation or alter result files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / 'artifacts/diagnostics'
OUT = ROOT / 'MUIRBENCH_RESULTS_INVENTORY_20260914.md'
lines = ['# MuirBench 已有结果汇总（2026-09-14）', '',
         '均为 Qwen3-VL-4B-Instruct 系列。分数单位为百分比。不同抽样、提示格式、像素处理、DeepStack 配置和干预不能混作同一排行榜。', '',
         '范围：当前工作区已保存的完整模型测评分数、历史结果和 MuirBench 专项诊断。未完成/已废弃结果单列；小规模 smoke 不视为模型准确率。没有数值的张量一致性检查不冒充 benchmark 成绩。', '']


def read(path):
    return json.loads(path.read_text())


def table(title, headers, rows, notes='', sources=()):
    lines.extend(['## ' + title, ''])
    if notes:
        lines.extend([notes, ''])
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + '|'.join('---' for _ in headers) + '|')
    for row in rows:
        lines.append('| ' + ' | '.join(str(v).replace('|', '\\|') for v in row) + ' |')
    lines.append('')
    for source in sources:
        lines.extend([f'来源：[{source.relative_to(ROOT)}]({source})', ''])


def fmt(v):
    return f'{v:.2f}' if isinstance(v, (int, float)) else str(v)


names = {'base': 'Base', 'embedding_adapter': 'Embedding Adapter + KL（全层，rank128）',
         'embedding_adapter_mixed': '混合数据训练 Embedding Adapter',
         'adapter_start7': '原生0–6，Adapter7–34，原生35（rank512）',
         'adapter_start16': '原生0–15，Adapter16–34，原生35（rank512）',
         'static_kl': 'Embedding Adapter + KL', 'recurrent_kl': 'Recurrent Adapter + KL',
         'sft': 'Embedding Adapter + SFT', 'opd': 'Embedding Adapter + OPD'}
p = DIAG / 'muir_random1000_seed42_matched_20260914/results.json'
q = DIAG / 'muir_late_adapter_random1000_20260914/results.json'
rows = []
for r in read(p) + read(q):
    assert r['benchmark'] == 'muirbench' and r['n'] == r['expected'] == 1000
    name = names.get(r['method'], r['method'].upper())
    if r['method'] == 'dart':
        name += f" 保留{100*r['retention']:g}%"
    rows.append([name, r['n'], fmt(r['accuracy'])])
table('1. 当前统一协议：随机1000条', ['方法', '题数', 'Accuracy'], rows,
      'seed42，从2600条无放回随机抽样；media_first_v1，原始图片处理，所有图片保留，DeepStack关闭，greedy最多8 tokens。两个后层 Adapter 是不同训练 checkpoint、rank512，不能当作只改变起始层的纯消融。其他方法没有当前协议下的结果时，不用历史成绩填补。', (p, q))

rows=[];sources=[]
for run, label, mode_names in [
    ('muir_images_as_video_20260914', '等分辨率图像/视频对照', {'matched_images':'512方形多图，单独输入', 'as_video':'相同图片按视频帧输入'}),
    ('muir_concat_images_20260914', '拼图对照', {'separate':'512方形带图号面板，单独输入', 'concat':'相同带图号面板，纵向拼成单图'})]:
    f=DIAG/run/'summary.json';sources.append(f)
    for mode,description in mode_names.items():
        rs=[r for r in read(f)['results'] if r.get('task','all')=='all' and r['mode']==mode]
        scores={r['method']:r['accuracy'] for r in rs};assert all(r['n']==1000 for r in rs)
        rows.append([label,description,1000,fmt(scores['base']),fmt(scores['embedding_adapter'])])
f=DIAG/'muir_separator_binding_20260914/summary.json';sources.append(f)
rows.append(['分隔符读取干预','只限制分隔符/图号 query 读取本图视觉 KV',1000,'未测',fmt(read(f)['local_separator_accuracy'])])
table('2. 同一随机1000条的输入/干预实验', ['实验','条件','题数','Base','Embedding Adapter'],rows,
      '全部关闭 DeepStack。视频与拼图实验分别有自己的匹配输入对照；不可把43.40与原协议40.70之差归因于拼接。分隔符干预改变 attention，不是普通评测协议。',sources)

p=DIAG/'adapter_nodeepstack_mediafirst_20260913/results.json';d=read(p)
rows=[[names[m],d[m]['muirbench']['samples'],fmt(d[m]['muirbench']['accuracy'])] for m in ['base','static_kl','recurrent_kl','sft','opd']]
q=DIAG/'muir_dart_mediafirst_20260914/results.json'
rows += [[f"DART 保留{100*r['retention']:g}%",r['n'],fmt(r['accuracy'])] for r in read(q) if r['benchmark']=='muirbench']
table('3. 旧样本集：原始文件前1000条，统一图片放前', ['方法','题数','Accuracy'],rows,
      '不是随机1000条；media_first_v1、DeepStack关闭。SFT原记录最多8 tokens为39.10；对159条截断回答延长到128 tokens后净多对10条，对应合并准确率40.10（其余841条沿用原记录）。', (p,q,DIAG/'muir_length_audit_20260914/summary.json'))

p=DIAG/'multimodal_baselines_nodeepstack_20260913/results.json';d=[r for r in read(p) if r['benchmark']=='muirbench']
base=next(r for r in d if r['method']=='base')
rows=[]
for m in ['fastv','dart','divprune','zoo','sparsevlm','visionzip']:
    scores={r['retention']:r['accuracy'] for r in d if r['method']==m}
    rows.append([m,fmt(scores[.2]),fmt(scores[.05])])
table('4. 六个 pruning baseline 的旧完整记录', ['方法','保留20%','保留5%'], rows,
      f"原始前1000条，每格1000条，DeepStack关闭；使用旧输入排布，不与第1/3节混比。同批 Base={base['accuracy']:.2f}。这里比例是脚本的视觉token保留预算，不是跨全部层求和后的计算占比。",(p,))

rows=[];sources=[]
for p in sorted((ROOT/'artifacts/experiments').rglob('muirbench/results.json')):
    d=read(p);sources.append(p)
    method='混合数据训练 Adapter' if 'qwen_mixed_adapter' in str(p) else names[p.parents[2].name]
    rows.append([method,d['total_samples'],fmt(d['teacher']['score_percent']),fmt(d['adapter']['score_percent'])])
table('5. 最早训练输出目录中的历史测评', ['训练版本','题数','配套Teacher/Base','Adapter'],rows,
      '旧协议存档，未统一为当前 no-DeepStack/修正后的推理和提示协议，不作为当前公平对比。混合训练历史结果是全2600条；其余为旧前1000条。',sources)

rows=[];sources=[]
for run,label in [
    ('muir_hf_adapter_inference_parity_random1000_20260914','随机1000，原始图片放前'),
    ('muir_hf_adapter_inference_parity_20260914','旧前1000，图片放前'),
    ('muir_hf_adapter_inference_parity_interleaved_20260914','旧前1000，图文交错'),
    ('muir_hf_adapter_inference_parity_eos_only_20260914','旧前1000，完整路径仅EOS停止')]:
    p=DIAG/run/'summary.json';d=read(p);sources.append(p)
    rows.append([label,d['samples'],fmt(d['candidate_accuracy']),fmt(d['reference_accuracy'])])
table('6. 同一 Adapter 的两套推理实现', ['输入协议','题数','快速路径','原生完整序列路径'],rows,
      '两边都是同一 Adapter，完整路径并不是Base：只是在HF原生层前替换相同的视觉memory。最新随机1000的18条BF16预测差异在FP32复核时全部一致，最大输出KL=2.54e-10。',sources)
rows=[];sources=[]
for run in ['muir_fullprecision_inference_20260914','muir_vision_backend_20260914','muir_source_pixels_20260914']:
    p=DIAG/run/'summary.json';d=read(p);sources.append(p)
    rows.extend([[run,k,d['samples'],fmt(v)] for k,v in d['accuracy'].items()])
table('7. 旧前1000条：数值精度与图像编码复核',['检查','条件','题数','Adapter Accuracy'],rows,
      '全部是旧前1000条，不是当前随机1000；不和第1节直接比较。',sources)

rows=[]
p=DIAG/'muir_candidate_isolation_20260914/summary.json'
for r in read(p)['results']:
    rows.append([names.get(r['method'],r['method']),r['condition'],fmt(r['sensitivity']),fmt(r['specificity']),fmt(r['top_candidate_accuracy'])])
table('8. 单图/多图候选判断（不是普通benchmark accuracy）',
      ['模型','条件','正确候选判Yes/73','错误候选判No/368','候选排名正确/73'],rows,
      '随机1000中的132条图片选择题，441个候选图；73条有正确图片、59条None。先分别问每个候选Yes/No，平均两种答案顺序的logit margin。single只给当前候选图片；multi_target给所有图并问指定候选。', (p,))

metric_names={'accuracy','sensitivity','specificity','balanced_accuracy','top_candidate_accuracy',
              'answerable_top_image_accuracy','forced_binary_accuracy','reject_all_unanswerable',
              'candidate_accuracy','reference_accuracy','original_accuracy','local_separator_accuracy',
              'accuracy_change_points'}


def collect(node, path='', count='未单列'):
    if isinstance(node,dict):
        count=node.get('n',node.get('samples',node.get('candidates',count)))
        labels=[f'{k}={node[k]}' for k in ('method','mode','condition','style','order','layout','task','subset') if k in node]
        here=path+(' ['+', '.join(labels)+']' if labels else '')
        for k,v in node.items():
            if k in metric_names and isinstance(v,(int,float)):
                denominator=node.get('positive_count',count) if k=='sensitivity' else node.get('negative_count',count) if k=='specificity' else node.get('answerable_scenes',count) if k in ('top_candidate_accuracy','answerable_top_image_accuracy') else node.get('unanswerable_scenes',count) if k=='reject_all_unanswerable' else count
                yield [here or 'root',k,denominator,fmt(v)]
            elif k=='accuracy' and isinstance(v,dict):
                for variant,score in v.items():
                    if isinstance(score,(int,float)):yield [here+'/'+variant,'accuracy',count,fmt(score)]
            elif isinstance(v,(dict,list)):
                yield from collect(v,here+'/'+k,count)
    elif isinstance(node,list):
        for i,v in enumerate(node):
            if isinstance(v,(dict,list)):yield from collect(v,path+f'/{i}',count)


lines.extend(['## 9. 所有 MuirBench 专项 summary 的评分明细', '',
              '下面按原始实验逐一列出，包括局部样本、oracle、排列/提示词、逐层替换、单图和读取干预。它们不是同一任务协议，不能把最高值当作当前Adapter的整体分数。`accuracy_change_points`是百分点变化，不是准确率；其余评分按原summary的百分比记录。分母未单列者请查看来源。', ''])
for p in sorted(DIAG.glob('muir*/**/summary.json')):
    rows=list(collect(read(p)))
    if rows:
        table(str(p.parent.relative_to(DIAG)),['条件/字段路径','指标','分母/规模','数值'],rows,sources=(p,))

p=DIAG/'multimodal_baselines_20260912/results.json'
rows=[[r['method'],f"{100*r['retention']:g}%",r['n'],fmt(r['accuracy'])] for r in read(p) if r['benchmark']=='muirbench' and r['n']]
table('10. 早期未完成/已替代记录（不纳入比较）',['方法','保留比例','已测题数','当时分数'],rows,
      '9月12日旧实现：FastV仅部分样本完成，后续实现/协议已修订。没有结果的空记录和smoke不算有效benchmark结果。', (p,))
OUT.write_text('\n'.join(lines)+'\n')
print(OUT)
print('lines',len(lines))
