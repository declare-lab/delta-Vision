"""Check one-read preparation against the original; time only the new path."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import torch
import src.model
from src.qwen_adapter_prepare import prepare_fa2_inputs
import paired_runtime_execution

original=src.model.prepare_qwen_embedding_adapter_inputs
seen=[]
checks=[]


def equal(a,b):
    if torch.is_tensor(a):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def prepare(model,adapter,ids,*args,**kwargs):
    actual=prepare_fa2_inputs(model,adapter,ids,*args,**kwargs)
    if not any(ids is old for old in seen):
        # First use is the untimed eager validation before any paired timing.
        expected=original(model,adapter,ids,*args,**kwargs)
        equal(actual,expected)
        seen.append(ids)
        checks.append(dict(input_shape=list(ids.shape),all_prepared_fields_bitwise_equal=True))
    return actual


if __name__=='__main__':
    src.model.prepare_qwen_embedding_adapter_inputs=prepare
    paired_runtime_execution.main()
    (ROOT/'test/results/deepstack_off_20260915/adapter_prepare_validation.json').write_text(json.dumps(checks,indent=2))
