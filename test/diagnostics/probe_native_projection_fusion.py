"""Reject projection fusion unless real prefill/decode intermediates are exact."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
from src.data import QwenBenchmarkDataset
from src.qwen_deepstack import disable_qwen_deepstack
from src.qwen_fused_norm import fused_qwen_rmsnorm
from src.benchmark_adapter_optimizations import MODEL, MANIFEST


def main():
    torch.set_num_threads(4)
    model, processor = load_baseline_model('base', MODEL, torch.bfloat16, 'cuda:0', 1., 'flash_attention_2')
    disable_qwen_deepstack(model)
    data = QwenBenchmarkDataset(str(MANIFEST), processor, 'videomme',
        data_root=str(ROOT/'data/benchmarks/videomme'), cache_dir=ROOT/'test/results/adapter_exact_20260915/inputs')
    handles, checks = [], []
    def install(layer, modules, label):
        weight = torch.cat([module.weight for module in modules], 0)
        splits = [m.weight.shape[0] for m in modules]
        values = []
        for part, module in enumerate(modules):
            def after(module, args, output, part=part):
                if part == 0:
                    values[:] = torch.nn.functional.linear(args[0], weight).split(splits, -1)
                expected = values[part]
                checks.append(dict(layer=layer, kind=label, part=part, length=output.shape[1],
                    exact=torch.equal(output, expected), max_diff=float((output-expected).abs().max())))
            handles.append(module.register_forward_hook(after))
    for index, layer in enumerate(model.model.language_model.layers):
        install(index, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj], 'qkv')
        install(index, [layer.mlp.gate_proj, layer.mlp.up_proj], 'gate_up')
        for name, norm in [('qnorm', layer.self_attn.q_norm), ('knorm', layer.self_attn.k_norm)]:
            def after_norm(module, args, output, index=index, name=name):
                candidate = fused_qwen_rmsnorm(args[0], module.weight, module.variance_epsilon, exact_reduction=False)
                checks.append(dict(layer=index, kind=name, length=output.shape[1],
                    exact=torch.equal(output, candidate), max_diff=float((output-candidate).abs().max())))
            handles.append(norm.register_forward_hook(after_norm))
    with torch.inference_mode():
        for index in [0, 333, 666]:
            inputs = _qwen_inputs_from_item(data[index], torch.device('cuda:0'))
            model.model.rope_deltas = None
            model.generate(**inputs, min_new_tokens=2, max_new_tokens=2, do_sample=False)
    for handle in handles: handle.remove()
    summary = {kind:dict(checks=sum(c['kind']==kind for c in checks),
        mismatches=sum(c['kind']==kind and not c['exact'] for c in checks),
        max_diff=max(c['max_diff'] for c in checks if c['kind']==kind)) for kind in {c['kind'] for c in checks}}
    output = ROOT/'test/results/video_base_20260915/projection_probe.json'
    output.write_text(json.dumps(dict(summary=summary, checks=checks), indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
