"""Real-model counterfactual checks using held-out Pixmo, never benchmark labels."""
import argparse
from pathlib import Path

import torch

from analysis.fig01b_hidden_prediction.initial_token_mlp_probe import Bank, PathHook, teacher, runtime, inputs_for, run_model
from analysis.fig01a_hidden_channels.visual_channel_native_cache import load_rows, dump_json, digest


def main(root):
    runtime()
    processor, model = teacher()
    path = root / 'best.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    bank = Bank(checkpoint['stats'], checkpoint['init']).cuda().eval()
    bank.load_state_dict(checkpoint['bank'])
    checkpoint_step = checkpoint['step']
    del checkpoint
    hook = PathHook(model)
    row = load_rows(root / 'validation.jsonl')[0]
    inp = inputs_for(processor, [row], model.device)
    results = {}
    with torch.no_grad():
        for mode, perturbation in [('hidden_mlp', 'layer_visual_output'), ('attn_mlp', 'native_attention_output')]:
            hook.begin(inp, mode, bank)
            ref = run_model(model, inp, hook, body=False).logits.float()
            perturb_calls = [0]
            handles = []
            def corrupt(module, args, output):
                h = output if torch.is_tensor(output) else output[0]
                if h.shape[1] == 1:
                    return
                changed = h.clone()
                for b, p in enumerate(hook.pos):
                    changed[b, p] = changed[b, p] + 100
                perturb_calls[0] += 1
                return changed if torch.is_tensor(output) else (changed, *output[1:])
            for li in (0, 17):
                module = hook.layers[li] if mode == 'hidden_mlp' else hook.layers[li].self_attn.o_proj
                handles.append(module.register_forward_hook(corrupt, prepend=mode == 'attn_mlp'))
            hook.begin(inp, mode, bank)
            intervened = run_model(model, inp, hook, body=False).logits.float()
            err = float((ref - intervened).abs().max())
            assert perturb_calls[0] == 2 and err == 0, (mode, err, perturb_calls)
            assert not hook.native and not hook.targets
            # Positive control: the very same corruption must alter a native run.
            hook.begin(inp, 'native')
            damaged_native = run_model(model, inp, hook, body=False).logits.float()
            for handle in handles:
                handle.remove()
            hook.begin(inp, 'native')
            native = run_model(model, inp, hook, body=False).logits.float()
            native_error = float((native - damaged_native).abs().max())
            assert native_error > 0, 'Perturbation positive control failed'
            results[mode] = dict(perturbation=perturbation, layers=[0, 17], student_max_logit_error=err,
                                 native_positive_control_max_logit_error=native_error)
    dump_json(root / 'student_counterfactual_validation.json', dict(checkpoint_step=checkpoint_step,
              test_source_sha256=digest(__file__), sample_image=row['image'], results=results))
    print(results, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
