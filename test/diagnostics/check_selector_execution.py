"""Compare original and optimized selector indices, synchronization and latency."""
import ast
import json
from pathlib import Path
import sys
import time
import statistics

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from baselines.multimodal_pruning_utils import visual_budget


def load(path, name):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(torch=torch, visual_budget=visual_budget)
    helper = next((n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_dart_neighbor_indices"), None)
    if helper is not None:
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), scope)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def elapsed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start), result


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    out = ROOT / "test/results/runtime_optimized_20260915"
    rows = []
    with torch.inference_mode():
        for method, name in [("divprune", "_divprune_select_tokens"), ("zoo", "_zoo_select_tokens"), ("dart", "dart_get_retained_image_token")]:
            relative = Path(f"baselines/{method}/qwen3_vl/modeling_qwen3_vl_{method}.py")
            old = load(out / "before" / relative, name)
            new = load(ROOT / relative, name)
            for count in [16, 177, 557, 2048]:
                features = torch.randn(count, 2560, device="cuda", dtype=torch.bfloat16)
                for retention in [.05, .2]:
                    keep = visual_budget(count, retention)
                    args = (features, keep)
                    if method == "zoo":
                        args = (features, torch.rand(count, device="cuda") if count != 16 else torch.ones(count, device="cuda"), keep)
                    if method == "dart":
                        hidden = torch.randn(1, count+80, 2560, device="cuda", dtype=torch.bfloat16)
                        keys = torch.randn(1, 8, count+80, 128, device="cuda", dtype=torch.bfloat16)
                        args = (dict(pivot_image_token=5,pivot_text_token=3,reduction_ratio=1-retention,target_retention=retention),hidden,keys,16,count)
                    expected, actual = old(*args), new(*args)
                    assert torch.equal(expected, actual), (method, count, retention)
                    times = []
                    for repeat in range(3):
                        pair = {}
                        for label, fn in ([("old",old),("new",new)] if repeat % 2 == 0 else [("new",new),("old",old)]):
                            pair[label], value = elapsed(lambda: fn(*args))
                            assert torch.equal(value, expected)
                        times.append(pair)
                    row = dict(method=method, visual_tokens=count, retention=retention, exact_indices=True,
                        old_ms=statistics.median(t["old"] for t in times), new_ms=statistics.median(t["new"] for t in times))
                    if count == 557 and retention == .2:
                        for label, fn in [("old",old),("new",new)]:
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                                fn(*args)
                            row[label+"_scalar_syncs"] = sum(e.count for e in prof.key_averages() if e.key == "aten::_local_scalar_dense")
                    rows.append(row)
                    (out / "selector_execution.json").write_text(json.dumps(rows,indent=2))
                    print(json.dumps(row),flush=True)


if __name__ == "__main__":
    main()
