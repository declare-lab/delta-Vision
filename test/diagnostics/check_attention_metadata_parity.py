"""Compare original and optimized real-model logits/cache under the same inputs."""
import json
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers import LogitsProcessor
from baselines.eval_baselines import load_baseline_model, configure_baseline, _qwen_inputs_from_item
from src.data import QwenBenchmarkDataset
from src.attention import optimize_qwen_attention_metadata


class Capture(LogitsProcessor):
    def __init__(self):
        self.logits = []
    def __call__(self, ids, scores):
        self.logits.append(scores.clone())
        return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["base", "fastv", "dart", "divprune", "zoo", "sparsevlm", "visionzip"])
    parser.add_argument("--samples", nargs="+", type=int, default=[0, 25, 125])
    parser.add_argument("--output", default="test/results/prefill_slowdown_investigation_20260915/parity.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = ROOT / args.output
    rows = []
    for name in args.methods:
        model, processor = load_baseline_model(name,
            str(Path(__file__).resolve().parents[2] / "model/Qwen3-VL-4B-Instruct"),
            torch.bfloat16, torch.device("cuda:0"), .05, "flash_attention_2")
        path = ROOT / "data/benchmarks/mmstar/mmstar_speedtest_200.jsonl"
        ds = QwenBenchmarkDataset(str(path), processor, "mmstar", data_root=str(path.parent), max_samples=200)
        optimization = optimize_qwen_attention_metadata(model)
        for index in args.samples:
            inputs = _qwen_inputs_from_item(ds[index], torch.device("cuda:0"))
            vis = inputs["mm_token_type_ids"][0].nonzero().flatten()
            configure_baseline(model, name, .05, int(vis[0]), len(vis))
            results, logits = [], []
            with torch.inference_mode():
                for enabled in [False, True]:
                    optimization.enabled = enabled
                    torch.manual_seed(42)
                    torch.cuda.manual_seed_all(42)
                    model.model.rope_deltas = None
                    cap = Capture()
                    result = model.generate(**inputs, max_new_tokens=8, do_sample=False, logits_processor=[cap],
                                            return_dict_in_generate=True)
                    results.append(result)
                    logits.append(cap.logits)
                same_ids = torch.equal(results[0].sequences, results[1].sequences)
                max_logit_diff = max(float((a-b).abs().max()) for a,b in zip(*logits))
                max_kv_diff = max(float((getattr(a, key)-getattr(b, key)).abs().max())
                    for a,b in zip(results[0].past_key_values.layers, results[1].past_key_values.layers)
                    for key in ["keys", "values"])
                rows.append(dict(method=name, sample_index=index, generated_ids_equal=same_ids,
                                 max_logit_diff=max_logit_diff, max_kv_diff=max_kv_diff,
                                 generated_ids=[r.sequences[0,inputs['input_ids'].shape[-1]:].tolist() for r in results],
                                 top5_first_logits=[{'ids':v[0][0].topk(5).indices.tolist(),'values':v[0][0].topk(5).values.tolist()} for v in logits]))
                print(json.dumps(rows[-1]), flush=True)
                out.write_text(json.dumps(rows, indent=2))
            del results, logits
        optimization.remove()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
