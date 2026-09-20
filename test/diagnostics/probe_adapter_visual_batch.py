"""Test batching the static adapter's independent visual norm/K/V branches."""
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.update(QWEN_VIDEO_SAMPLING='full_timestamp_v1', QWEN_VIDEO_NUM_FRAMES='8', HF_HUB_DISABLE_PROGRESS_BARS='1')
import torch
from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item
from src.benchmark_adapter_optimizations import MODEL, CHECKPOINT, MANIFEST
from src.benchmark_prefill import build_qwen_fast_adapter_prefill
from src.data import QwenBenchmarkDataset
from src.model import load_qwen_embedding_adapter_checkpoint, build_qwen_initial_context, prepare_qwen_embedding_adapter_inputs
from src.qwen_adapter_kernels import exact_rope
from src.qwen_deepstack import disable_qwen_deepstack

OUTPUT = ROOT/'test/results/adapter_max_20260915/visual_batch_probe.json'
INPUT_CACHE = ROOT/'test/results/qwen3vl4b_embedding_m4multi64k_video64k_rank128_4000_20260915_step3000_8gpu/videomme/processed/videomme'


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): output = fn()
    graph.replay()
    return graph, output


def norm(x, weight, eps):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return y.to(x.dtype) * weight


def main():
    torch.set_num_threads(4)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    model, processor = load_baseline_model('base', MODEL, torch.bfloat16, torch.device('cuda:0'), 1., 'flash_attention_2')
    disable_qwen_deepstack(model)
    adapter, _ = load_qwen_embedding_adapter_checkpoint(str(CHECKPOINT), model.model.language_model, torch.device('cuda:0'), torch.bfloat16)
    settings = SimpleNamespace(last_logits_only=True, attn_implementation='flash_attention_2', cuda_graph=True,
        cuda_graph_context=True, compile_verify=True, compile_max_diff=0., cuda_graph_warmup=3,
        adapter_decode_cache_mode='fast', adapter_exact_optimizations=True)
    unused = build_qwen_fast_adapter_prefill(model, adapter, settings)
    layers = model.model.language_model.layers
    dataset = QwenBenchmarkDataset(str(MANIFEST), processor, 'videomme',
        data_root=str(ROOT/'data/benchmarks/videomme'), cache_dir=str(INPUT_CACHE))
    rows = []
    with torch.inference_mode():
        norm_weights = torch.stack([l.input_layernorm.weight for l in layers])[:, None, None]
        knorm_weights = torch.stack([l.self_attn.k_norm.weight for l in layers])[:, None, None, None]
        key_weights = torch.stack([l.self_attn.k_proj.weight for l in layers])
        value_weights = torch.stack([l.self_attn.v_proj.weight for l in layers])
        kv_weights = torch.cat([key_weights, value_weights], dim=1)
        for index in [0, 69, 114, 423, 756]:
            inputs = _qwen_inputs_from_item(dataset[index], torch.device('cuda:0'))
            hidden, positions = build_qwen_initial_context(model, inputs)
            prepared = prepare_qwen_embedding_adapter_inputs(model, adapter, inputs['input_ids'], inputs['attention_mask'], inputs['mm_token_type_ids'], hidden, positions)
            memories = adapter.all_visual_memories_batched(prepared['visual_memory'])
            L, B, N, H = memories.shape
            D = layers[0].self_attn.head_dim
            K = key_weights.shape[1]
            embeddings = tuple(t.expand(L, -1, -1) for t in prepared['visual_position_embeddings'])
            def serial():
                norms, keys, values = [], [], []
                for l, vm in zip(layers, memories.unbind(0)):
                    x = l.input_layernorm(vm)
                    key = l.self_attn.k_norm(l.self_attn.k_proj(x).view(B, N, -1, D)).transpose(1,2)
                    key = exact_rope(key, prepared['visual_position_embeddings'])
                    norms.append(x)
                    keys.append(key)
                    values.append(l.self_attn.v_proj(x).view(B, N, -1, D).transpose(1,2))
                return torch.stack(norms), torch.stack(keys), torch.stack(values)
            def batched(merged=False):
                x = norm(memories, norm_weights, layers[0].input_layernorm.variance_epsilon)
                if merged:
                    kv = torch.bmm(x.view(L, N, H), kv_weights.transpose(1,2))
                    key, value = (part.contiguous() for part in kv.split(K, dim=-1))
                else:
                    key = torch.bmm(x.view(L, N, H), key_weights.transpose(1,2))
                    value = torch.bmm(x.view(L, N, H), value_weights.transpose(1,2))
                key = norm(key.view(L, B, N, -1, D), knorm_weights, layers[0].self_attn.k_norm.variance_epsilon)
                key = exact_rope(key[:,0].transpose(1,2), embeddings).unsqueeze(1)
                return x, key, value.view(L, B, N, -1, D).transpose(2,3)
            functions = dict(serial=serial, batched=batched, merged=lambda:batched(True))
            graphs, outputs = {}, {}
            for label, fn in functions.items():
                graphs[label], outputs[label] = capture(fn)
            row = dict(index=index, visual_tokens=N, parity={})
            for label in ['batched','merged']:
                row['parity'][label] = [dict(equal=torch.equal(a,b), unequal=int(a.ne(b).sum()),
                    max_diff=float((a-b).abs().max())) for a,b in zip(outputs['serial'], outputs[label])]
            trials = []
            for repetition in range(15):
                trial = {}
                for label in (list(graphs) if repetition%2==0 else list(graphs)[::-1]):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    graphs[label].replay()
                    torch.cuda.synchronize()
                    trial[label] = (time.perf_counter()-start)*1000
                trials.append(trial)
            row['median_ms'] = {l:statistics.median(t[l] for t in trials) for l in graphs}
            rows.append(row)
            OUTPUT.write_text(json.dumps(rows,indent=2))
            print(json.dumps(row), flush=True)
            del graphs, outputs
    model._adapter_decode_graph_runner.remove()


if __name__=='__main__':main()
