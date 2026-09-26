"""CPU checks for the vLLM export; GPU integration records live in artifacts."""
import json
from pathlib import Path
import tempfile
import unittest
import argparse
import hashlib

import torch
from safetensors.torch import load_file
from src.vllm_adapter import VisualAdapter, export


class ExportTests(unittest.TestCase):
    def test_qwen35_norms_match_transformers_rounding(self):
        from types import SimpleNamespace
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm, Qwen3_5RMSNormGated
        from src.vllm_adapter import _hf_gemma_norm, _hf_gated_norm
        torch.manual_seed(44)
        x, residual, gate = [torch.randn(7,128,dtype=torch.bfloat16) for _ in range(3)]
        norm = Qwen3_5RMSNorm(128).to(torch.bfloat16)
        torch.nn.init.normal_(norm.weight)
        shim = SimpleNamespace(weight=norm.weight,variance_epsilon=norm.eps)
        self.assertTrue(torch.equal(norm(x),_hf_gemma_norm(shim,x)))
        result, carried = _hf_gemma_norm(shim,x,residual)
        self.assertTrue(torch.equal(norm(x+residual),result))
        self.assertTrue(torch.equal(carried,x+residual))
        gated = Qwen3_5RMSNormGated(128).to(torch.bfloat16)
        torch.nn.init.normal_(gated.weight)
        shim = SimpleNamespace(weight=gated.weight,eps=gated.variance_epsilon)
        self.assertTrue(torch.equal(gated(x,gate),_hf_gated_norm(shim,x,gate)))

    def test_recurrent_export_and_qwen35_layout(self):
        from src.model import QwenEmbeddingAdapter
        for family in ('qwen3_vl', 'qwen3_5'):
            for mode in ('embedding_adapter', 'recurrent_embedding_adapter'):
                with self.subTest(family=family, mode=mode), tempfile.TemporaryDirectory() as td:
                    torch.manual_seed(44)
                    source = QwenEmbeddingAdapter(hidden_size=16, num_layers=3,
                        num_heads=2, head_dim=8, mode=mode, visual_adapter_rank=4).eval()
                    for up in source.visual_adapter_up:
                        torch.nn.init.normal_(up.weight, std=.1)
                    root = Path(td); base = root/'base'; base.mkdir()
                    (base/'config.json').write_text(json.dumps(dict(model_type=family,
                        text_config=dict(hidden_size=16,num_hidden_layers=3),
                        vision_config=dict(deepstack_visual_indexes=[]))))
                    state = source.state_dict()
                    if family == 'qwen3_5':
                        state = {k.removeprefix('visual_adapter_'): v for k,v in state.items()}
                    ckpt = root/'adapter.pt'
                    torch.save(dict(state_dict=state,args=dict(output_mode=mode),
                        config=dict(architecture='static_embedding_adapter' if mode == 'embedding_adapter' else mode)),ckpt)
                    output = export(base,ckpt,root/'export')
                    target = VisualAdapter(16,3,4)
                    target.load_state_dict(load_file(str(output/'adapter.weights')))
                    x = torch.randn(2,5,16)
                    expected = source.all_visual_memories_batched(x)
                    current = x
                    for i in range(3):
                        actual = target(current,i)
                        torch.testing.assert_close(actual,expected[i],atol=1e-6,rtol=1e-6)
                        if mode == 'recurrent_embedding_adapter':
                            current = actual
                    if mode == 'recurrent_embedding_adapter':
                        self.assertFalse(torch.allclose(current,target(x,2)))

    def test_export_preserves_trained_residual_function(self):
        from src.model import QwenEmbeddingAdapter
        torch.manual_seed(44)
        source = QwenEmbeddingAdapter(hidden_size=16, num_layers=3,
            num_heads=2, head_dim=8, mode='embedding_adapter', visual_adapter_rank=4)
        for up in source.visual_adapter_up:
            torch.nn.init.normal_(up.weight)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / 'base'; base.mkdir()
            cfg = dict(model_type='qwen3_vl', architectures=['Qwen3VLForConditionalGeneration'],
                text_config=dict(hidden_size=16, num_hidden_layers=3),
                vision_config=dict(deepstack_visual_indexes=[0, 1]))
            (base / 'config.json').write_text(json.dumps(cfg))
            (base / 'model.safetensors').write_bytes(b'base weights are linked, not rewritten')
            ckpt = root / 'adapter.pt'
            torch.save(dict(state_dict=source.state_dict(),args=dict(output_mode='embedding_adapter')),ckpt)
            output = export(base,ckpt,root / 'export')
            target = VisualAdapter(16,3,4)
            target.load_state_dict(load_file(str(output / 'adapter.weights')))
            x = torch.randn(2,5,16)
            for i in range(3):
                self.assertTrue(torch.equal(source.visual_memory_for_layer(x,i),target(x,i)))
            for sample in x:
                batched = target.all_memories(sample)
                expected = torch.stack([target(sample, i) for i in range(3)])
                torch.testing.assert_close(batched, expected, rtol=1e-5, atol=1e-5)
            self.assertTrue((output / 'model.safetensors').is_symlink())
            exported = json.loads((output / 'config.json').read_text())
            self.assertEqual(exported['vision_config']['deepstack_visual_indexes'],[])
            self.assertEqual(json.loads((base / 'config.json').read_text()),cfg)
            self.assertEqual(exported['delta_vision_adapter']['rank'],4)
            with self.assertRaises(FileExistsError):
                export(base,ckpt,output)
            saved=torch.load(ckpt,weights_only=False)
            saved['args']['output_mode']='unknown';torch.save(saved,ckpt)
            with self.assertRaisesRegex(ValueError,'Unsupported'):
                export(base,ckpt,root / 'invalid')
            self.assertFalse((root / 'invalid').exists())


def capture_logits(model):
    """Worker-local validation instrumentation, never enabled in the model plugin."""
    model._validation_logits = []
    original = model.compute_logits

    def wrapped(hidden):
        logits = original(hidden)
        model._validation_logits.append(logits.detach().cpu())
        return logits

    model.compute_logits = wrapped
    model._validation_embeddings = []
    model._validation_positions = []
    model._validation_pixels = []
    model._validation_vision = {}
    model._validation_mixed_seen = False
    model._validation_layers = {}
    if model.config.model_type == 'qwen3_5':
        gdn=model.language_model.model.layers[0].linear_attn
        for kind,module in [('qkvz',gdn.in_proj_qkvz),('ba',gdn.in_proj_ba),('gated_norm',gdn.norm)]:
            def trace_gdn(module,args,result,_key=f'gdn0.{kind}'):
                value=result[0] if isinstance(result,tuple) else result
                model._validation_layers.setdefault(_key,[]).append(value.detach().cpu())
                if _key.endswith('gated_norm'):
                    model._validation_layers.setdefault('gdn0.core',[]).append(args[0].detach().cpu())
            module.register_forward_hook(trace_gdn)
        attn=model.language_model.model.layers[3].self_attn
        for kind,module in [('qkv',attn.qkv_proj),('q_norm',attn.q_norm),('k_norm',attn.k_norm)]:
            def trace_fa(module,args,result,_key=f'fa3.{kind}'):
                value=result[0] if isinstance(result,tuple) else result
                model._validation_layers.setdefault(_key,[]).append(value.detach().cpu())
            module.register_forward_hook(trace_fa)
        def trace_rope(module,args,result):
            for key,value in zip(('q','k'),result):
                model._validation_layers.setdefault(f'fa3.rotary_{key}',[]).append(value.detach().cpu())
        attn.rotary_emb.register_forward_hook(trace_rope)
        def trace_attn(module,args,result):
            model._validation_layers.setdefault('fa3.attn',[]).append(result.detach().cpu())
        attn.attn.register_forward_hook(trace_attn)
        for i, layer in enumerate(model.language_model.model.layers):
            for kind, module in [('input_norm',layer.input_layernorm),
                                 ('post_norm',layer.post_attention_layernorm),
                                 ('mlp',layer.mlp)]:
                def trace(module,args,result,_key=f'{i}.{kind}'):
                    value=result[0] if isinstance(result,tuple) else result
                    model._validation_layers.setdefault(_key,[]).append(value.reshape(-1,value.shape[-1])[-8:].detach().cpu())
                module.register_forward_hook(trace)
            module=layer.linear_attn if layer.layer_type=='linear_attention' else layer.self_attn
            def trace_mixer(module,args,kwargs,result,_key=f'{i}.mixer'):
                value=kwargs['output']
                model._validation_layers.setdefault(_key,[]).append(value[-8:].detach().cpu())
            module.register_forward_hook(trace_mixer,with_kwargs=True)
    for key, module in [('patch',model.visual.patch_embed),
                        ('norm1',model.visual.blocks[0].norm1),
                        ('qkv',model.visual.blocks[0].attn.qkv),
                        ('proj',model.visual.blocks[0].attn.proj),
                        ('block0',model.visual.blocks[0])]:
        def hook(module,args,result,_key=key):
            value=result[0] if isinstance(result,tuple) else result
            model._validation_vision[_key]=value.reshape(-1,value.shape[-1])[:32].detach().cpu()
        module.register_forward_hook(hook)
    def pixel_hook(module, args, kwargs):
        pixels = kwargs.get('x', args[0] if args else None)
        grid = kwargs.get('grid_thw', args[1] if len(args)>1 else None)
        model._validation_pixels.append((pixels.detach().cpu(),torch.as_tensor(grid).cpu()))
    model.visual.register_forward_pre_hook(pixel_hook,with_kwargs=True)
    original_forward = model.forward

    def wrapped_forward(*args, **kwargs):
        from vllm.forward_context import get_forward_context
        md=get_forward_context().attn_metadata
        if isinstance(md,dict):
            model._validation_mixed_seen |= any(getattr(v,'num_prefills',0)>0 and
                getattr(v,'num_decodes',0)>0 for v in md.values())
        h = kwargs.get('inputs_embeds')
        mask = model._adapter_mm_mask
        if h is not None and mask is not None and bool(mask.any()):
            model._validation_embeddings.append(h[:mask.numel()][mask.to(h.device)].detach().cpu())
            model._validation_positions.append(kwargs['positions'].detach().cpu())
        return original_forward(*args, **kwargs)

    model.forward = wrapped_forward
    return True


def validate_vllm(args):
    import os
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_PLUGINS'] = 'delta_vision'
    # Trusted local test callbacks only; no server is launched, no persistent setting.
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.export)
    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines()[:2]]
    cases, requests = [], []
    for i, row in enumerate(rows):
        path = str((Path(args.image_root) / row['image']).resolve())
        with Image.open(path) as im:
            image = im.convert('RGB').copy()
        content = [dict(type='image'),dict(type='text',text=row['question'])]
        if i == 1:
            content.insert(0,dict(type='text',text='Look carefully at this image.'))
        prompt = processor.apply_chat_template([dict(role='user',content=content)],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        requests.append(dict(prompt=prompt,multi_modal_data=dict(image=image)))
        cases.append(dict(image=path,prompt=prompt,row=row))
    llm = LLM(model=args.export,dtype='bfloat16',tensor_parallel_size=1,
        enforce_eager=True,enable_prefix_caching=False,enable_chunked_prefill=False,
        async_scheduling=False,max_model_len=4096,max_num_batched_tokens=4096,
        max_num_seqs=2,gpu_memory_utilization=0.18,kv_cache_memory_bytes=1024**3,skip_mm_profiling=True,
        mamba_ssm_cache_dtype='float32',
        limit_mm_per_prompt={'image':1,'video':0},seed=44,
        attention_config={'backend':'FLASH_ATTN','flash_attn_version':2},
        gdn_prefill_backend=args.gdn_prefill_backend)
    llm.apply_model(capture_logits)
    sampling = SamplingParams(temperature=0,max_tokens=8)
    batch = llm.generate(requests,sampling,use_tqdm=False)
    def dump_batch(model):
        torch.save(model._validation_logits,output/'vllm_batch_logits.pt')
        return True
    llm.apply_model(dump_batch)
    for i,(case,request) in enumerate(zip(cases,requests)):
        llm.reset_mm_cache()
        llm.llm_engine.reset_encoder_cache()
        llm.apply_model(lambda m: m._validation_logits.clear())
        llm.apply_model(lambda m: m._validation_embeddings.clear())
        llm.apply_model(lambda m: m._validation_positions.clear())
        llm.apply_model(lambda m: m._validation_pixels.clear())
        llm.apply_model(lambda m: m._validation_vision.clear())
        llm.apply_model(lambda m: m._validation_layers.clear())
        result = llm.generate([request],sampling,use_tqdm=False)[0]
        def dump_captured(model):
            # Save in the worker: nested tensors are not losslessly serialized by
            # vLLM's RPC codec in every case (tuples may contain memoryviews).
            for kind in ('logits','embeddings','positions','pixels','vision','layers'):
                torch.save(getattr(model,f'_validation_{kind}'),output/f'vllm_{kind}_{i}.pt')
            return len(model._validation_logits)
        llm.apply_model(dump_captured)
        case.update(prompt_token_ids=result.prompt_token_ids,
            token_ids=list(result.outputs[0].token_ids),text=result.outputs[0].text,
            batch_token_ids=list(batch[i].outputs[0].token_ids))
    if json.loads((Path(args.export)/'config.json').read_text())['model_type']=='qwen3_5':
        llm.apply_model(lambda m: setattr(m,'_validation_mixed_seen',False))
        llm.enqueue([requests[0]],sampling,use_tqdm=False)
        finished=llm.llm_engine.step()  # Existing request now has a nonempty hybrid cache.
        llm.enqueue([requests[1]],sampling,use_tqdm=False)
        mixed=sorted([r for r in finished if r.finished]+
                     llm.wait_for_completion(use_tqdm=False),key=lambda r:int(r.request_id))
        assert len(mixed)==len(cases)
        seen=any(llm.apply_model(lambda m: m._validation_mixed_seen))
        for case,result in zip(cases,mixed):
            assert result.prompt_token_ids==case['prompt_token_ids']
            case['mixed_token_ids']=list(result.outputs[0].token_ids)
            case['mixed_prefill_decode_seen']=seen
    (output / 'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    (output / 'export_path.txt').write_text(str(Path(args.export).resolve()))
    (output / 'protocol.json').write_text(json.dumps(dict(
        samples=len(cases),benchmark='RealWorldQA diagnostic only, not accuracy evaluation',
        dtype='bfloat16',deepstack=False,thinking=False,attention='FA2',
        mamba_ssm_cache_dtype='float32',
        gdn_prefill_backend='fla.chunk_gated_delta_rule' if json.loads((Path(args.export)/'config.json').read_text())['model_type']=='qwen3_5' else None,
        native_gdn_backend_requested=args.gdn_prefill_backend or 'auto',
        full_attention_kernel='HF FA2 fixed-length, paged KV gathered for decode' if json.loads((Path(args.export)/'config.json').read_text())['model_type']=='qwen3_5' else 'native vLLM FA2',
        eager=True,fast_path=False,chunked_prefill=False,prefix_caching=False,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in ('src/vllm_adapter.py','test/diagnostics/test_vllm_adapter.py')}),indent=2)+'\n')
    print([(c['text'],c['token_ids']==c['batch_token_ids']) for c in cases],flush=True)


def logit_metrics(a, b, token, step):
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()[:a.numel()]
    la, lb = a.double().log_softmax(-1), b.double().log_softmax(-1)
    return dict(step=step,argmax_matches=int(a.argmax())==token,
        hf_argmax=int(a.argmax()),vllm_token=token,
        logit_rmse=float((a-b).square().mean().sqrt()),
        centered_logit_rmse=float(((a-a.mean())-(b-b.mean())).square().mean().sqrt()),
        logit_cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0)),
        kl_hf_to_vllm=float((la.exp()*(la-lb)).sum()))


def finish_report(output, report):
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    steps=[s for c in report for s in c['steps']]
    status=dict(samples=len(report),generated_steps=len(steps),
        batch_generation_matches=all(c['batch_matches'] for c in report),
        hf_greedy_matches=all(s['argmax_matches'] for s in steps),
        pixels_equal=all(c['pixel_max_abs']==0 for c in report),
        max_kl=max(s['kl_hf_to_vllm'] for s in steps),
        min_cosine=min(s['logit_cosine'] for s in steps),
        numerical_gate=dict(kl_lt=1e-3,cosine_gt=.999))
    status['passed']=all(status[k] for k in ('batch_generation_matches','hf_greedy_matches','pixels_equal')) and (
        status['max_kl']<1e-3 and status['min_cosine']>.999)
    mixed=[c for c in report if c.get('mixed_prefill_decode_seen') is not None]
    if mixed:
        status['mixed_prefill_decode_seen']=all(c['mixed_prefill_decode_seen'] for c in mixed)
        status['mixed_generation_matches']=all(c['mixed_generation_matches'] for c in mixed)
        status['passed'] &= status['mixed_prefill_decode_seen'] and status['mixed_generation_matches']
    (output/'validation_status.json').write_text(json.dumps(status,indent=2)+'\n')
    assert status['passed'], status


def validate_reference35(args, config):
    from PIL import Image
    from src.qwen35 import load_model, initial_context
    output = Path(args.output); spec = config['delta_vision_adapter']
    device = torch.device('cuda:0'); torch.set_num_threads(4)
    processor, model, adapter, controller = load_model(
        dict(model_path=spec['base_model'],rank=spec['rank']),device)
    saved = torch.load(spec['source_checkpoint'],map_location='cpu',weights_only=False)
    adapter.load_state_dict(saved['state_dict'],strict=True)
    adapter.eval()
    layers={}
    gdn=model.model.language_model.layers[0].linear_attn
    for kind,module in [('qkv',gdn.in_proj_qkv),('z',gdn.in_proj_z),('b',gdn.in_proj_b),('a',gdn.in_proj_a),('gated_norm',gdn.norm)]:
        def trace_gdn(module,args,result,_key=f'gdn0.{kind}'):
            layers.setdefault(_key,[]).append(result.detach().cpu())
            if _key.endswith('gated_norm'):
                layers.setdefault('gdn0.core',[]).append(args[0].detach().cpu())
        module.register_forward_hook(trace_gdn)
    attn=model.model.language_model.layers[3].self_attn
    for kind,module in [('q',attn.q_proj),('k',attn.k_proj),('v',attn.v_proj),('q_norm',attn.q_norm),('k_norm',attn.k_norm)]:
        def trace_fa(module,args,result,_key=f'fa3.{kind}'):
            layers.setdefault(_key,[]).append(result.detach().cpu())
        module.register_forward_hook(trace_fa)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    original_rope=modeling.apply_rotary_pos_emb
    active=[False]
    attn.register_forward_pre_hook(lambda *_: active.__setitem__(0,True))
    attn.register_forward_hook(lambda *_: active.__setitem__(0,False))
    def trace_rope(*args,**kwargs):
        result=original_rope(*args,**kwargs)
        if active[0]:
            for key,value in zip(('q','k'),result):
                layers.setdefault(f'fa3.rotary_{key}',[]).append(value.transpose(1,2).detach().cpu())
        return result
    modeling.apply_rotary_pos_emb=trace_rope
    for i,layer in enumerate(model.model.language_model.layers):
        for kind,module in [('input_norm',layer.input_layernorm),
                            ('post_norm',layer.post_attention_layernorm),('mlp',layer.mlp),
                            ('mixer',layer.linear_attn if layer.block_type=='linear_attention' else layer.self_attn)]:
            def trace(module,args,result,_key=f'{i}.{kind}'):
                value=result[0] if isinstance(result,tuple) else result
                layers.setdefault(_key,[]).append(value.reshape(-1,value.shape[-1])[-8:].detach().cpu())
            module.register_forward_hook(trace)
    vision={}
    for key,module in [('patch',model.model.visual.patch_embed),
                       ('norm1',model.model.visual.blocks[0].norm1),
                       ('qkv',model.model.visual.blocks[0].attn.qkv),
                       ('proj',model.model.visual.blocks[0].attn.proj),
                       ('block0',model.model.visual.blocks[0])]:
        def hook(module,args,result,_key=key):
            vision[_key]=result.reshape(-1,result.shape[-1])[:32].detach().cpu()
        module.register_forward_hook(hook)
    if spec['mode'] == 'recurrent_embedding_adapter':
        # Explicit diagnostic reference for recurrent checkpoints; no static
        # training artifact is silently reinterpreted as recurrent.
        def recurrent(embeddings):
            states=[]
            with torch.autocast('cuda',dtype=embeddings.dtype):
                for down,up in zip(adapter.down,adapter.up):
                    embeddings=embeddings+up(torch.nn.functional.silu(down(embeddings)))
                    states.append(embeddings)
            return tuple(states)
        adapter.forward=recurrent
    report=[]
    with torch.inference_mode():
        for i,case in enumerate(json.loads((output/'cases.json').read_text())):
            with Image.open(case['image']) as im:
                image=im.convert('RGB').copy()
            inputs=processor(text=[case['prompt']],images=[image],return_tensors='pt').to(device)
            assert inputs['input_ids'][0].tolist()==case['prompt_token_ids']
            context=initial_context(model,inputs)
            vision_metrics={}
            if (output/f'vllm_vision_{i}.pt').exists():
                traced=torch.load(output/f'vllm_vision_{i}.pt',weights_only=True)
                vision_metrics={key:float((vision[key].float()-value.float()).square().mean().sqrt())
                                for key,value in traced.items()}
            h,pos=context['inputs_embeds'],context['position_ids']
            mask=inputs['mm_token_type_ids'].eq(1)
            pixels=torch.load(output/f'vllm_pixels_{i}.pt',weights_only=True)
            assert len(pixels)==1
            pixel,grid=pixels[0]
            assert torch.equal(grid,inputs['image_grid_thw'].cpu())
            pixel_error=float((pixel.float()-inputs['pixel_values'].to(pixel.dtype).float().cpu()).abs().max())
            vp=torch.load(output/f'vllm_positions_{i}.pt',weights_only=True)
            assert len(vp)==1 and torch.equal(vp[0],pos[:,0].cpu()), 'M-RoPE positions differ'
            ve=torch.load(output/f'vllm_embeddings_{i}.pt',weights_only=True)
            assert len(ve)==1
            ve=ve[0].to(h); he=h[mask]
            other=torch.load(output/f'vllm_logits_{i}.pt',weights_only=True)
            assert len(other)==len(case['token_ids'])
            layers.clear()
            with controller.activate('adapter',mask):
                hidden=model.model.language_model(**dict(context,use_cache=True))
                cache=hidden.past_key_values
                logits=model.lm_head(hidden.last_hidden_state[:,-1:])
                steps=[]
                for step,token in enumerate(case['token_ids']):
                    steps.append(logit_metrics(logits[0,-1],other[step][0],token,step))
                    # Native cached text decode continues both GDN and full-attn state.
                    result=model(input_ids=torch.tensor([[token]],device=device),
                        past_key_values=cache,use_cache=True,logits_to_keep=1)
                    logits,cache=result.logits,result.past_key_values
            torch.save(layers,output/f'hf_layers_{i}.pt')
            same_context=dict(context,inputs_embeds=h.clone())
            same_context['inputs_embeds'][mask]=ve
            with controller.activate('adapter',mask):
                same=model.model.language_model(**dict(same_context,use_cache=True))
                cache=same.past_key_values
                logits=model.lm_head(same.last_hidden_state[:,-1:])
                shared_steps=[]
                for step,token in enumerate(case['token_ids']):
                    shared_steps.append(logit_metrics(logits[0,-1],other[step][0],token,step))
                    result=model(input_ids=torch.tensor([[token]],device=device),
                        past_key_values=cache,use_cache=True,logits_to_keep=1)
                    logits,cache=result.logits,result.past_key_values
            report.append(dict(case=i,text=case['text'],steps=steps,
                mixed_generation_matches=case.get('mixed_token_ids')==case['token_ids'] if 'mixed_token_ids' in case else None,
                mixed_prefill_decode_seen=case.get('mixed_prefill_decode_seen'),
                vision_stage_rmse=vision_metrics,
                batch_matches=case['token_ids']==case['batch_token_ids'],
                prompt_tokens=len(case['prompt_token_ids']),pixel_max_abs=pixel_error,
                embedding_rmse=float((ve.float()-he.float()).square().mean().sqrt()),
                embedding_cosine=float(torch.nn.functional.cosine_similarity(ve.float(),he.float(),dim=-1).mean()),
                shared_embedding_prefill_kl=shared_steps[0]['kl_hf_to_vllm'],
                shared_embedding_steps=shared_steps))
            print(report[-1],flush=True)
    finish_report(output,report)


def validate_reference(args):
    from PIL import Image
    from src import model as m
    output=Path(args.output)
    config=json.loads((Path((output/'export_path.txt').read_text())/'config.json').read_text())
    if config['model_type'] == 'qwen3_5':
        return validate_reference35(args, config)
    spec=config['delta_vision_adapter']; device=torch.device('cuda:0')
    torch.set_num_threads(4)
    processor,model=m.load_frozen_qwen3vl(spec['base_model'],torch.bfloat16,device,'flash_attention_2')
    model._adapter_attention_implementation='flash_attention_2'
    adapter,_=m.load_qwen_embedding_adapter_checkpoint(spec['source_checkpoint'],
        model.model.language_model,device,torch.bfloat16)
    cases=json.loads((output/'cases.json').read_text()); report=[]
    with torch.inference_mode():
        for i,case in enumerate(cases):
            with Image.open(case['image']) as im:
                image=im.convert('RGB').copy()
            inputs=processor(text=[case['prompt']],images=[image],return_tensors='pt').to(device)
            assert inputs['input_ids'][0].tolist()==case['prompt_token_ids'], 'Input tokens differ'
            h,pos=m.build_qwen_initial_context(model,inputs)
            visual=inputs['mm_token_type_ids'][0].bool()
            exported_embeddings=torch.load(output/f'vllm_embeddings_{i}.pt',
                map_location='cpu',weights_only=True)
            assert len(exported_embeddings)==1, 'Expected one freshly captured image embedding'
            visual_metrics={}
            pixel_path=output/f'vllm_pixels_{i}.pt'
            if pixel_path.exists():
                pixels=torch.load(pixel_path,map_location='cpu',weights_only=True)
                assert len(pixels)==1, 'Encoder cache must be cleared before each single-request check'
                if pixels:
                    pixel,grid=pixels[0]
                    assert torch.equal(grid,inputs['image_grid_thw'].cpu())
                    visual_metrics['pixel_max_abs']=float((pixel.float()-inputs['pixel_values'].to(pixel.dtype).float().cpu()).abs().max())
                    visual_metrics['pixel_dtype']=str(pixel.dtype)
            vp=torch.load(output/f'vllm_positions_{i}.pt',map_location='cpu',weights_only=True)
            assert len(vp)==1 and torch.equal(vp[0],pos[:,0].cpu()), 'M-RoPE positions differ'
            if exported_embeddings:
                ve=torch.cat(exported_embeddings,dim=0).to(device=device,dtype=h.dtype)
                he=h[0,visual]
                assert ve.shape==he.shape
                visual_metrics.update(embedding_rmse=float((ve.float()-he.float()).square().mean().sqrt()),
                    embedding_cosine=float(torch.nn.functional.cosine_similarity(ve.float(),he.float(),dim=-1).mean()))
            prepared=m.prepare_qwen_embedding_adapter_inputs(model,adapter,inputs['input_ids'],
                inputs['attention_mask'],inputs['mm_token_type_ids'],h,pos)
            logits,_,cache=m.qwen_embedding_adapter_prefill_cache_prepared(model,adapter,
                **prepared,retain_prefix_states=False,logits_to_keep=1)
            other=torch.load(output/f'vllm_logits_{i}.pt',map_location='cpu',weights_only=True)
            assert len(other)==len(case['token_ids'])
            native=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float().cpu()
            native_lp=native.double().log_softmax(-1)
            vl_lp=other[0][0].double()[:native.numel()].log_softmax(-1)
            native_kl=float((native_lp.exp()*(native_lp-vl_lp)).sum())
            if exported_embeddings:
                same_h=h.clone(); same_h[0,visual]=ve
                same_prepared=m.prepare_qwen_embedding_adapter_inputs(model,adapter,
                    inputs['input_ids'],inputs['attention_mask'],inputs['mm_token_type_ids'],same_h,pos)
                same_logits,_,_=m.qwen_embedding_adapter_prefill_cache_prepared(model,adapter,
                    **same_prepared,retain_prefix_states=False,logits_to_keep=1)
                same_lp=same_logits[0,-1].double().cpu().log_softmax(-1)
                visual_metrics['shared_embedding_prefill_kl']=float((same_lp.exp()*(same_lp-vl_lp)).sum())
            steps=[]
            for step,token in enumerate(case['token_ids']):
                a=logits[0,-1].float().cpu(); b=other[step][0].float()[:a.numel()]
                la,lb=a.double().log_softmax(-1),b.double().log_softmax(-1)
                steps.append(dict(step=step,argmax_matches=int(a.argmax())==token,
                    hf_argmax=int(a.argmax()),vllm_token=token,
                    logit_rmse=float((a-b).square().mean().sqrt()),
                    centered_logit_rmse=float(((a-a.mean())-(b-b.mean())).square().mean().sqrt()),
                    logit_cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0)),
                    kl_hf_to_vllm=float((la.exp()*(la-lb)).sum())))
                logits,cache=m.qwen_embedding_adapter_decode_step(model,adapter,
                    torch.tensor([[token]],device=device),cache,logits_to_keep=1)
            report.append(dict(case=i,text=case['text'],steps=steps,
                batch_matches=case['token_ids']==case['batch_token_ids'],
                prompt_tokens=len(case['prompt_token_ids']),native_to_vllm_prefill_kl=native_kl))
            report[-1].update(visual_metrics)
            print(report[-1],flush=True)
    # Empirical BF16 regression tolerances for these diagnostic cases, not a
    # guarantee of benchmark-level accuracy equivalence on unseen inputs.
    finish_report(output,report)


if __name__ == '__main__':
    import sys
    if '--vllm' in sys.argv or '--reference' in sys.argv:
        parser=argparse.ArgumentParser(description='GPU vLLM/HF functional comparison')
        mode=parser.add_mutually_exclusive_group(required=True)
        mode.add_argument('--vllm',action='store_true');mode.add_argument('--reference',action='store_true')
        parser.add_argument('--export');parser.add_argument('--data');parser.add_argument('--image-root')
        parser.add_argument('--gdn-prefill-backend',choices=('triton','flashinfer'))
        parser.add_argument('--output',required=True)
        args=parser.parse_args()
        if args.vllm and not all((args.export,args.data,args.image_root)):
            parser.error('--vllm requires --export, --data, and --image-root')
        (validate_vllm if args.vllm else validate_reference)(args)
    else:
        unittest.main()
