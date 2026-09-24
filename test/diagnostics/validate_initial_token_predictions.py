"""Check selected native targets against the old full path, then worst batches."""
import json
from pathlib import Path
import time

import torch

from analysis.fig01b_hidden_prediction.initial_token_prediction_probe import Capture, LAYERS, KEYS, OLD
from analysis.fig01b_hidden_prediction.initial_token_mlp_probe import PathHook, Bank, teacher, runtime, inputs_for, run_model, TRAIN
from analysis.fig01a_hidden_channels.visual_channel_native_cache import load_rows, dump_json


def main():
    root = Path('artifacts/diagnostics/initial_token_predictions_l15_17_20260916')
    runtime()
    processor, model = teacher()
    rows = load_rows(root/'train.jsonl')
    val = load_rows(root/'validation.jsonl')
    inp = inputs_for(processor, [val[0]], model.device)
    old = PathHook(model)
    old.begin(inp, 'capture', cache_native=False)
    with torch.no_grad():
        native = run_model(model, inp, old).last_hidden_state.detach().clone()
    initial = old.initial.clone()
    reference = {f'{kind}_{l}': old.targets[f'{kind}_{l if kind=="cross" else l-1}'].clone()
                 for kind in ('cross','hidden') for l in LAYERS}
    for handle in old.handles:
        handle.remove()
    del old
    hook = Capture(model)
    x, y, _ = hook.collect(inp, stop_early=False)
    errors = {'initial':float((x.float()-initial.float()).abs().max())}
    errors.update({k:float((v-y[k]).abs().max()) for k,v in reference.items()})
    assert max(errors.values()) == 0, errors
    full = {k:v.clone() for k,v in y.items()}
    rounding = {str(l):float((y[f'delta_{l}']-hook.raw_attention[l].float()).abs().max()) for l in LAYERS}
    _, short, _ = hook.collect(inp)
    assert all(torch.equal(short[k], full[k]) for k in KEYS)
    hook.begin(inp); hook.stop_early = False
    model.model.rope_deltas = None
    with torch.no_grad():
        captured = model.model(**inp, use_cache=False, return_dict=True).last_hidden_state
    error = float((native.float()-captured.float()).abs().max())
    assert error == 0, error
    print('PATH', errors, 'unchanged_native', error, 'delta_rounding_vs_o_proj', rounding, flush=True)
    del native, captured, reference, full, initial, short, x, y, inp
    oldstats = torch.load(OLD/'normalization.pt',weights_only=False,map_location='cpu')
    # Statistics here are only for a memory/gradient test; this bank is discarded.
    stats=dict(input=oldstats['input'],targets={})
    for l in LAYERS:
        stats['targets'][f'cross_{l}']=oldstats['targets'][f'cross_{l}']
        stats['targets'][f'hidden_{l}']=oldstats['targets'][f'hidden_{l-1}']
        stats['targets'][f'delta_{l}']=oldstats['targets'][f'cross_{l}']
    bank=Bank(stats,'zero',keys=KEYS).cuda()
    opt=torch.optim.AdamW(bank.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.01,fused=True)
    areas=json.loads(Path(str(TRAIN)+'.pixel_areas.json').read_text())['areas']
    candidates=[]
    for step in range(2000):
        for rank in range(8):
            start=(step*32+rank*4)%len(rows)
            pixels=sum(areas[rows[(start+j)%len(rows)]['source_row']] for j in range(4))
            candidates.append((pixels,step,rank))
    _, maxstep,maxrank=max(candidates)
    results=[]
    for iteration,(step,rank) in enumerate(((636,2),(maxstep,maxrank))):
        tick=time.time(); torch.cuda.reset_peak_memory_stats()
        start=(step*32+rank*4)%len(rows)
        batch=[rows[(start+j)%len(rows)] for j in range(4)]
        x,y,sizes=hook.collect(inputs_for(processor,batch,model.device))
        opt.zero_grad(set_to_none=True)
        loss,_=bank(x,y,sizes);loss.backward()
        grads={k:dict(down=float(h.down.weight.grad.norm()),up=float(h.up.weight.grad.norm())) for k,h in bank.heads.items()}
        assert all(v['up']>0 and (iteration==0 or v['down']>0) for v in grads.values())
        grad=torch.nn.utils.clip_grad_norm_(bank.parameters(),1.)
        assert torch.isfinite(loss) and torch.isfinite(grad)
        assert all(p.grad is None for p in model.parameters())
        opt.step();torch.cuda.synchronize()
        record=dict(step=step+1,original_rank=rank,visual_tokens=sizes,loss=float(loss),grad_norm=float(grad),
                    gradients=grads,seconds=time.time()-tick,peak_memory_bytes=torch.cuda.max_memory_allocated())
        results.append(record);print('STRESS',record,flush=True)
    hook.close()
    dump_json(root/'verification.json',dict(exact_old_target_errors=errors,early_stop_exact=True,
              unchanged_native_max_error=error,delta_rounding_vs_o_proj=rounding,memory_stress=results,
              memory_test_bank_discarded=True,attention='flash_attention_2',deepstack='off',
              unit_tests=dict(command=".venv/bin/python -m unittest discover -s test/diagnostics -p 'test_initial_token*'",passed=6)))


if __name__=='__main__':
    main()
