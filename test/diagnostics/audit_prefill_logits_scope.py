"""Check whether computing all prompt logits explains the remaining base-time gap.

This is a separate scope diagnostic, never a replacement denominator in the table.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.benchmarking.common.prefill import benchmark, configure_torch_runtime
from src.data import QwenBenchmarkDataset
from baselines.eval_baselines import load_baseline_model, _qwen_inputs_from_item


def main():
    configure_torch_runtime()
    device = torch.device("cuda:0")
    model, processor = load_baseline_model("base", "/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct",
                                            torch.bfloat16, device, 1., "flash_attention_2")
    data = ROOT / "data/benchmarks/mmstar/mmstar_speedtest_200.jsonl"
    dataset = QwenBenchmarkDataset(str(data), processor, "mmstar", data_root=str(data.parent), max_samples=200)
    rows = []
    with torch.inference_mode():
        for j, index in enumerate(range(0, 200, 25)):
            inputs = _qwen_inputs_from_item(dataset[index], device)
            row = {"subset_index": index, "tokens": inputs["input_ids"].shape[-1]}
            for keep in ([1, 0] if j % 2 == 0 else [0, 1]):
                def forward():
                    model.model.rope_deltas = None
                    return model(**inputs, logits_to_keep=keep).logits
                row["last_logits_ms" if keep else "all_logits_ms"] = benchmark(forward, warmup=2, n_runs=5) * 1000
            row["all_over_last"] = row["all_logits_ms"] / row["last_logits_ms"]
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {"purpose": "Diagnostic of logits scope, historical logits setting unknown", "rows": rows,
              "ratio_of_sums": sum(r["all_logits_ms"] for r in rows) / sum(r["last_logits_ms"] for r in rows)}
    (ROOT / "test/results/prefill_original_mmstar200_20260915/logits_scope.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
