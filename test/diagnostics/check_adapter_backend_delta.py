"""Reproduce every changed adapter answer by switching only its attention plan."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.model import load_frozen_qwen3vl, load_qwen_embedding_adapter_checkpoint
from src.model import build_qwen_initial_context, prepare_qwen_embedding_adapter_inputs, qwen_embedding_adapter_prefill_cache_prepared
from src.data import QwenBenchmarkDataset
from baselines.eval_baselines import _qwen_inputs_from_item


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    processor, model = load_frozen_qwen3vl("/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct",
        torch.bfloat16, device, "flash_attention_2")
    adapter, _ = load_qwen_embedding_adapter_checkpoint(str(ROOT / "artifacts/experiments/pixmo_adapter_comparison/static_recurrent_sft_opd_20260911/static_kl/checkpoints/qwen_embedding_adapter_step2000.pt"),
        model.model.language_model, device, torch.bfloat16)
    path = ROOT / "data/benchmarks/mmstar/mmstar_speedtest_200.jsonl"
    dataset = QwenBenchmarkDataset(str(path), processor, "mmstar", data_root=str(path.parent), max_samples=200)
    old = json.loads((ROOT / "test/results/prefill_decode_corrected_20260915/adapter.details.json").read_text())["predictions"]
    new = json.loads((ROOT / "test/results/all_fa2_20260915/adapter200.details.json").read_text())["predictions"]
    changed = [i for i,(a,b) in enumerate(zip(old,new)) if a["adapter_text"] != b["adapter_text"]]
    model._adapter_attention_implementation = "flash_attention_2"
    rows = []
    with torch.inference_mode():
        for index in [0, *changed]:
            inputs = _qwen_inputs_from_item(dataset[index], device)
            hidden, positions = build_qwen_initial_context(model, inputs)
            prepared = prepare_qwen_embedding_adapter_inputs(model, adapter, inputs["input_ids"], inputs["attention_mask"],
                inputs["mm_token_type_ids"], hidden, positions)
            logits_fa2 = qwen_embedding_adapter_prefill_cache_prepared(model, adapter, **prepared)[0].flatten().float()
            prepared["attention_plan"] = None
            logits_sdpa = qwen_embedding_adapter_prefill_cache_prepared(model, adapter, **prepared)[0].flatten().float()
            a, b = int(logits_sdpa.argmax()), int(logits_fa2.argmax())
            text_a, text_b = processor.tokenizer.decode([a]), processor.tokenizer.decode([b])
            assert text_a == old[index]["adapter_text"], (index,text_a,old[index]["adapter_text"])
            assert text_b == new[index]["adapter_text"], (index,text_b,new[index]["adapter_text"])
            rows.append(dict(index=index, sdpa=text_a, fa2=text_b, reproduced_both_saved_answers=True,
                max_abs_logit_difference=float((logits_fa2-logits_sdpa).abs().max()),
                sdpa_winner_margin=float(logits_sdpa[a]-logits_sdpa[b]),
                fa2_winner_margin=float(logits_fa2[b]-logits_fa2[a])))
            print(json.dumps(rows[-1]), flush=True)
    (ROOT / "test/results/all_fa2_20260915/adapter_backend_delta.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
