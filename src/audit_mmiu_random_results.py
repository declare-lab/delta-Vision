"""Rescore immutable generated outputs without treating mentions as answers."""
from collections import Counter
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/diagnostics/mmiu_random1000_seed42_all_methods_20260914'


def extract_answer(text, choices):
    text=text.rsplit('</think>',1)[-1].strip()
    text=re.sub(r'^```(?:\w+)?\s*\n?|\n?```$', '', text).strip()
    letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(choices)]
    # A complete letter response, or a response beginning with a labelled
    # option. Never search arbitrary mentions/English articles in the body.
    first_line=text.splitlines()[0].strip().strip('*` ') if text else ''
    match=re.fullmatch(r'[\[(]?([A-Z])[\])]?[.:：]?',first_line)
    if match and match[1] in letters:
        return match[1]
    match=re.match(r'^\s*[*(\[]*([A-Z])[)\].:：]\s+',text)
    if match and match[1] in letters:
        return match[1]
    # Only explicit decisions; "options A, B, C, D" is not a decision.
    matches=list(re.finditer(
        r'(?:\b(?:final\s+answer|answer|correct\s+(?:answer|option|choice))|答案|选项)'
        r'\s*(?:is|是|:|：)\s*[*\s(\[]*([A-Z])(?:[)\].,:：*\s]|$)',text,re.I))
    if matches:
        letter=matches[-1][1].upper()
        if letter in letters:
            return letter
    # One clearly labelled answer line inside an explanation is valid. A list
    # of multiple candidate labels is not itself an answer.
    labelled=re.findall(r'^\s*[*\s]*([A-Z])\s*[.:：)]\s+\S',text,re.M)
    if len(labelled)==1 and labelled[0] in letters:
        return labelled[0]
    # Exact textual option is permitted; substring matches in a refusal are not.
    normalized=text.casefold().strip(' .\n\t')
    for i,choice in enumerate(choices):
        if normalized==str(choice).casefold().strip(' .\n\t'):
            return letters[i]
    return None


def audit():
    manifest=[json.loads(l) for l in (OUT/'mmiu_random1000.jsonl').open()]
    rows=[];changes=[]
    for p in sorted(OUT.glob('*_shard*.jsonl')):
        for line in p.open():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue  # Last in-progress line only; final count is validated below.
            sample=manifest[r['index']]
            assert r['source_index']==sample['index'] and r['gold']==sample['answer']
            prediction=extract_answer(r['text'],sample['choices'])
            new=dict(r,legacy_prediction=r['prediction'],legacy_score=r['score'],
                prediction=prediction,score=float(prediction is not None and prediction==r['gold']),
                invalid=prediction is None,scoring_version='explicit_answer_only_v1')
            if new['prediction']!=r['prediction'] or new['score']!=r['score']:
                changes.append(new)
            rows.append(new)
    assert len(rows)==len({(r['method'],r['retention'],r['index']) for r in rows})
    inputs={}
    for r in rows:
        i=r['index']
        if i in inputs:assert inputs[i]==r['input_sha256']
        inputs[i]=r['input_sha256']
    status=json.loads((OUT/'status.json').read_text())
    complete=status['state']=='complete'
    if complete:assert len(rows)==17000
    result=[]
    for method,retention in sorted({(r['method'],r['retention']) for r in rows}):
        rs=[r for r in rows if (r['method'],r['retention'])==(method,retention)]
        valid=[r for r in rs if r['annotation_valid']]
        if complete:
            assert len(rs)==1000 and len(valid)==998
        tasks={t:[r for r in valid if r['task']==t] for t in sorted({r['task'] for r in valid})}
        result.append(dict(method=method,retention=retention,n=len(rs),
            accuracy=100*sum(r['score'] for r in rs)/len(rs),
            valid_n=len(valid),valid_accuracy=100*sum(r['score'] for r in valid)/len(valid) if valid else None,
            invalid_predictions=sum(r['invalid'] for r in rs),
            legacy_accuracy=100*sum(r['legacy_score'] for r in rs)/len(rs),
            per_task={t:dict(n=len(g),accuracy=100*sum(r['score'] for r in g)/len(g)) for t,g in tasks.items()}))
    payload=dict(complete=complete,scoring_version='explicit_answer_only_v1',records=len(rows),results=result)
    for name,data in [('audited_results.json',payload),('answer_parsing_changes.json',changes)]:
        target=OUT/name;temp=target.with_suffix('.tmp')
        temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(target)
    if complete:
        (OUT/'audited_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    for r in result:
        print(r['method'],r['retention'],r['n'],round(r['accuracy'],2),'valid',round(r['valid_accuracy'],2))
    print('records',len(rows),'complete',complete,'parser changes',len(changes))


if __name__=='__main__':
    options=['first','second','third','fourth']
    assert extract_answer('A',options)=='A'
    assert extract_answer('Answer: B.',options)=='B'
    assert extract_answer('None of the options A, B, C, D are valid. Answer: None of the above.',options) is None
    assert extract_answer('The images show a chair.',options) is None
    assert extract_answer('A chair is visible.',options) is None
    assert extract_answer('The correct order is:\n\nA: [2, 1, 4, 3]',options)=='A'
    audit()
