"""Real-weight loss/gradient/update/logit parity against pre-refactor snapshots.

Run each family in a fresh process. This intentionally uses the retained original
code, not a second alias of the new entrypoint, as the reference.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def reference(root, file, name):
    expected = json.loads((root/'sha256.json').read_text())[file]
    assert hashlib.sha256((root/file).read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location('src._reference_'+name, root/file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # Historical snapshots retain their original hashes. Translate only their
    # imports to the consolidated public modules; leave numerical code intact.
    moved = {
        'qwen35_embedding': 'qwen35', 'qwen35_experiment': 'qwen35',
        'qwen_deepstack': 'model_setup', 'evaluation_sampling': 'data',
        'fixed_multimodal_inputs': 'data', 'document_metrics': 'benchmarks',
        'benchmark_video_sampling': 'video', 'video_benchmark_inputs': 'video',
        'qwen_adapter_fa2': 'attention', 'qwen_adapter_prepare': 'attention',
        'qwen_attention_metadata': 'attention',
    }
    class Imports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module:
                prefix = 'src.' if node.module.startswith('src.') else ''
                old = node.module.removeprefix(prefix) if prefix else node.module
                if (prefix or node.level) and old in moved:
                    node.module = prefix + moved[old]
            return node
    tree = Imports().visit(ast.parse((root/file).read_text()))
    exec(compile(tree, str(root/file), 'exec'), module.__dict__)
    return module


def compare(a, b):
    import torch
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    error = float((a.detach().float()-b.detach().float()).abs().max())
    assert torch.equal(a, b), f'Numerical drift: max_abs={error}'
    return error


def compare_parameters(a, b, gradients=False):
    left, right = dict(a.named_parameters()), dict(b.named_parameters())
    assert left.keys() == right.keys()
    errors = []
    for k in left:
        x, y = left[k], right[k]
        if gradients:
            x, y = x.grad, y.grad
            assert (x is None) == (y is None), k
            if x is None:
                continue
        errors.append(compare(x, y))
    assert errors
    return max(errors)


def collect_logits(model, call):
    import torch
    values = []
    def capture(module, inputs, output):
        # Full vocabulary at the final position, for prefill and each decode.
        values.append(output[:, -1].detach().cpu().clone())
    handle = model.lm_head.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            result = call()
    finally:
        handle.remove()
    assert values
    return result, values


def check_generation(model, first, second):
    a, x = collect_logits(model, first)
    b, y = collect_logits(model, second)
    assert a == b, (a, b)
    assert len(x) == len(y)
    error = max(compare(i, j) for i, j in zip(x, y))
    return dict(logit_steps=len(x), max_abs_logit_error=error, outputs_identical=True)


def qwen(args):
    import torch
    from src.run import backend
    from src.model_setup import create_qwen_adapter, load_qwen_embedding_adapter_checkpoint
    old_train = reference(args.reference, 'src/train.py', 'train')
    old_eval = reference(args.reference, 'src/eval_benchmarks.py', 'eval')
    old_model = reference(args.reference, 'src/model.py', 'model')
    train, evaluate = backend('train', 'qwen'), backend('eval', 'qwen')
    from src.model import prepare_qwen3vl_batch_inputs
    device, dtype = torch.device('cuda:0'), torch.bfloat16
    processor, model = train.load_frozen_qwen3vl(args.model, dtype, device, 'flash_attention_2')
    torch.manual_seed(44)
    a = old_model.QwenEmbeddingAdapter.from_language_model(model.model.language_model,
        mode='embedding_adapter', visual_adapter_rank=128).to(device=device, dtype=dtype)
    torch.manual_seed(44)
    b = create_qwen_adapter(model.model.language_model, mode='embedding_adapter', rank=128, device=device, dtype=dtype)
    initial = compare_parameters(a, b)
    opt_a = torch.optim.AdamW(a.parameters(), lr=5e-5, betas=(.9,.95))
    opt_b = torch.optim.AdamW(b.parameters(), lr=5e-5, betas=(.9,.95))
    options = ['--model-kind','qwen','--output-dir',str(args.output.parent), '--loss-normalization','token',
               '--supervision-loss','distill','--temperature','2','--kl-topk','1024','--lambda-logit','2']
    cfg = train.parse_args(options)
    # Reference parser predates explicit argv support.
    previous = sys.argv
    try:
        sys.argv = ['reference'] + options
        old_cfg = old_train.parse_args()
    finally:
        sys.argv = previous
    assert vars(cfg) == vars(old_cfg)
    rows = []
    with Path(args.data).open() as handle:
        for _ in range(2):
            rows.append(json.loads(next(handle)))
    results = []
    for i, row in enumerate(rows):
        inputs, ids, mask, _ = prepare_qwen3vl_batch_inputs(processor,[row],Path(args.image_root),device,include_answers=True)
        a.train(); b.train(); opt_a.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
        torch.manual_seed(44+i)
        loss_a = old_train.compute_qwen_loss_for_prepared_inputs(old_cfg,model,a,inputs,ids,mask,36)[0]
        loss_a.backward()
        torch.manual_seed(44+i)
        loss_b = train.compute_qwen_loss_for_prepared_inputs(cfg,model,b,inputs,ids,mask,36)[0]
        loss_b.backward()
        report = dict(step=i+1, old_loss=float(loss_a.detach()), new_loss=float(loss_b.detach()),
            loss_max_abs=compare(loss_a,loss_b), gradient_max_abs=compare_parameters(a,b,True))
        opt_a.step(); opt_b.step()
        report['updated_parameter_max_abs'] = compare_parameters(a,b)
        results.append(report)
        print(json.dumps(report), flush=True)
    a.eval(); b.eval()
    # Exercise the common checkpoint construction used by evaluation as well.
    ckpt = args.output.parent/'parity_temporary_adapter.pt'
    torch.save(dict(state_dict=b.state_dict(), args=dict(output_mode='embedding_adapter',visual_adapter_rank=128)),ckpt)
    try:
        c, _ = load_qwen_embedding_adapter_checkpoint(ckpt,model.model.language_model,device,dtype)
        checkpoint_error = compare_parameters(b,c)
    finally:
        ckpt.unlink(missing_ok=True)
    inputs, _, _, _ = prepare_qwen3vl_batch_inputs(processor,[rows[0]],Path(args.image_root),device,include_answers=False)
    native = check_generation(model,
        lambda: old_eval.generate_teacher_qwen(model,processor,**inputs,max_new_tokens=8),
        lambda: evaluate.generate_teacher_qwen(model,processor,**inputs,max_new_tokens=8))
    adapter = check_generation(model,
        lambda: old_eval.generate_adapter_qwen_decode_cache(model,processor,a,inputs,8,decode_cache_mode='fast'),
        lambda: evaluate.generate_adapter_qwen_decode_cache(model,processor,c,inputs,8,decode_cache_mode='fast'))
    return dict(initial_parameter_max_abs=initial, checkpoint_parameter_max_abs=checkpoint_error,
                training=results, native=native, adapter=adapter)


def qwen35(args):
    # Bind the exact installed hybrid kernels before importing Transformers/Torch.
    sys.path.insert(0, str(ROOT/'artifacts/dependencies/qwen35_python'))
    import torch
    from src.run import backend
    from src import qwen35 as new
    from src.qwen35 import VisualAdapterController
    old = reference(args.reference,'src/qwen35_experiment.py','experiment35')
    cfg = dict(model_path=args.model,rank=128)
    torch.manual_seed(44)
    processor, model, a, controller = new.load_model(cfg, torch.device('cuda:0'))
    if args.fixed_weight_only:
        checkpoint=ROOT/'artifacts/experiments/qwen35_pixmo/qwen35_4b_embedding128_pixmo2000_20260920_064827/checkpoints/qwen35_embedding_adapter_step2000.pt'
        a.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        result=[]
        with Path(args.data).open() as handle:
            for index in range(2):
                row=json.loads(next(handle))
                inputs,plen=new.prepare_inputs(processor,row,args.image_root,torch.device('cuda:0'),training=True)
                context=new.initial_context(model,inputs)
                with torch.no_grad():
                    idx,prob=old.teacher_targets(model,context,plen,inputs['input_ids'][:,plen:],1024,2.)
                losses=[]
                for fn in (old.student_loss,backend('train','qwen35').student_loss):
                    with controller.activate('adapter',inputs['mm_token_type_ids'].eq(1),checkpoint_layers=True):
                        loss=fn(model,context,plen,idx,prob,2.)
                        losses.append(loss.detach().cpu())
                        del loss
                result.append(dict(sample=index,old_loss=float(losses[0]),new_loss=float(losses[1]),
                                   loss_max_abs=compare(*losses)))
        controller.close()
        return dict(checkpoint=str(checkpoint),fixed_weight_training_loss=result)
    # Preserve both original and shared construction paths, including initial RNG.
    b = __import__('copy').deepcopy(a)
    initial = compare_parameters(a,b)
    engine_train, engine_eval = backend('train','qwen35'), backend('eval','qwen35')
    assert engine_train.load_model is engine_eval.load_model
    opt_a = torch.optim.AdamW(a.parameters(),lr=5e-5,betas=(.9,.95),fused=True)
    opt_b = torch.optim.AdamW(b.parameters(),lr=5e-5,betas=(.9,.95),fused=True)
    control_adapters=[__import__('copy').deepcopy(a) for _ in range(2)]
    control_optimizers=[torch.optim.AdamW(c.parameters(),lr=5e-5,betas=(.9,.95),fused=True) for c in control_adapters]
    rows=[]
    with Path(args.data).open() as f:
        for _ in range(2): rows.append(json.loads(next(f)))
    results=[]
    for i,row in enumerate(rows):
        inputs, plen = new.prepare_inputs(processor,row,args.image_root,torch.device('cuda:0'),training=True)
        context=new.initial_context(model,inputs)
        with torch.no_grad():
            idx, prob=old.teacher_targets(model,context,plen,inputs['input_ids'][:,plen:],1024,2.)
            idx_new,prob_new=engine_train.teacher_targets(model,context,plen,inputs['input_ids'][:,plen:],1024,2.)
        compare(idx,idx_new);compare(prob,prob_new)
        # GDN fused backward is not bitwise deterministic. Measure the old-path
        # repeat spread along independent old-code optimizer trajectories.
        def backward(adapter, loss_fn):
            adapter.zero_grad(set_to_none=True)
            controller.adapter=adapter
            with controller.activate('adapter',inputs['mm_token_type_ids'].eq(1),checkpoint_layers=True):
                loss=loss_fn(model,context,plen,idx,prob,2.)
                loss.backward()
            grads=torch.cat([p.grad.detach().float().flatten().cpu() for p in adapter.parameters() if p.grad is not None])
            return loss.detach().cpu(),grads
        loss_a,grad_a=backward(a,old.student_loss)
        controls=[backward(c,old.student_loss) for c in control_adapters]
        loss_b,grad_b=backward(b,engine_train.student_loss)
        repeat_max=max(float((g-grad_a).abs().max()) for _,g in controls)
        repeat_l2=max(float((g-grad_a).double().norm()) for _,g in controls)
        grad_diff=float((grad_b-grad_a).abs().max())
        grad_l2=float((grad_b-grad_a).double().norm())
        # The bound is derived from old-vs-old execution, with an explicit
        # twofold repeat-spread allowance. Report relative errors explicitly.
        relative=grad_l2/max(float(grad_a.double().norm()),1e-30)
        loss_diff=float((loss_a-loss_b).abs())
        assert loss_diff <= max(1e-6,2*max(float((l-loss_a).abs()) for l,_ in controls)),loss_diff
        assert grad_l2 <= max(1e-7,2*repeat_l2),(grad_l2,repeat_l2,relative)
        report=dict(step=i+1,old_loss=float(loss_a),new_loss=float(loss_b),loss_max_abs=loss_diff,
            gradient_max_abs=grad_diff,gradient_relative_l2=relative,
            old_repeat_gradient_max_abs=repeat_max,old_repeat_gradient_l2=repeat_l2,
            old_repeat_gradient_relative_l2=repeat_l2/max(float(grad_a.double().norm()),1e-30),
            new_gradient_error_l2=grad_l2,gradient_within_repeat_tolerance=True)
        opt_a.step();opt_b.step()
        for optimizer in control_optimizers:optimizer.step()
        report['old_repeat_loss_max_abs']=max(float((l-loss_a).abs()) for l,_ in controls)
        report['old_repeat_updated_parameter_max_abs']=max(float((x-y).detach().abs().max()) for c in control_adapters for x,y in zip(a.parameters(),c.parameters()))
        report['updated_parameter_max_abs']=max(float((x-y).detach().abs().max()) for x,y in zip(a.parameters(),b.parameters()))
        results.append(report);print(json.dumps(report),flush=True)
    # Inference comparison requires identical weights; optimizer roundoff from
    # nondeterministic backward must not masquerade as an inference-path change.
    b.load_state_dict(a.state_dict(),strict=True)
    inputs,_=new.prepare_inputs(processor,rows[0],args.image_root,torch.device('cuda:0'))
    # Generation using shared evaluation helper; compare full-vocabulary logits
    # to the pre-refactor helper, not just argmax or extracted answers.
    from src.benchmarks import get_benchmark_spec
    spec=get_benchmark_spec('mmstar');config={'evaluation_generation':{'max_new_tokens':8,'do_sample':False,'unfinished_response':'invalid_zero'}}
    def generate(module,method,adapter):
        controller.adapter=adapter
        with controller.activate(method,inputs['mm_token_type_ids'].eq(1)):
            return module.generate_evaluation_answer(model,processor,inputs,rows[0],spec,config,max_new_tokens=8)
    native=check_generation(model,lambda:generate(old,'native',a),lambda:generate(engine_eval,'native',b))
    adapter=check_generation(model,lambda:generate(old,'adapter',a),lambda:generate(engine_eval,'adapter',b))
    controller.close()
    return dict(initial_parameter_max_abs=initial,training=results,native=native,adapter=adapter)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--family',choices=['qwen','qwen35'],required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--image-root',required=True)
    p.add_argument('--fixed-weight-only',action='store_true')
    args=p.parse_args()
    if args.family=='qwen35':sys.path.insert(0,str(ROOT/'artifacts/dependencies/qwen35_python'))
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    result=(qwen if args.family=='qwen' else qwen35)(args)
    result.update(family=args.family,passed=True,model=args.model,device=torch.cuda.get_device_name(),
        precision='bfloat16',attention='flash_attention_2',deepstack=False,
        samples=2,optimizer_steps=0 if args.fixed_weight_only else 2,reference=str(args.reference))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS',args.output,flush=True)


if __name__=='__main__':main()
