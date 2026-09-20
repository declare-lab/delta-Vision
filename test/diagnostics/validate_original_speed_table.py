"""Reconcile the original-entry-point table with every underlying sample."""
import hashlib
import json
import math
from pathlib import Path
import sys


def main():
    table = Path(sys.argv[1]).resolve()
    rows = json.loads(table.read_text())
    protocol = json.loads(table.with_suffix(".protocol.json").read_text())
    assert protocol["protocol"] == "original_entry_points"
    assert protocol["status"] == "complete"
    assert protocol["max_new_tokens"] == 8
    assert protocol["cuda_graph"] and protocol["cuda_graph_context"]
    assert protocol["structured_answer_early_stop"]
    assert len(rows) == 16, len(rows)
    data = Path(protocol["data_path"])
    assert hashlib.sha256(data.read_bytes()).hexdigest() == protocol["dataset_sha256"]
    assert hashlib.sha256(Path(protocol["checkpoint"]).read_bytes()).hexdigest() == protocol["checkpoint_sha256"]
    report = []

    def close(actual, expected):
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)

    for row in rows:
        assert row["samples"] == 200
        source = Path(row["source"])
        if row["reference_group"] == "adapter":
            detail = json.loads(source.with_suffix(".details.json").read_text())
            predictions = detail["predictions"]
            side = "teacher" if row["method"] == "qwen_base" else "adapter"
            total_key, prefill_key = side + "_total_s", side + "_prefill_s"
            kv_key, flops_key = side + "_kv_cache_mb", side + "_prefill_flops"
            ref = next(r for r in rows if r["reference_group"] == "adapter" and r["method"] == "qwen_base")
            if side == "adapter":
                for item in predictions:
                    close(item["adapter_total_s"], item["adapter_prefill_s"] + item["adapter_decode_s"])
                    close(item["adapter_prefill_kv_cache_mb"], item["teacher_kv_cache_mb"])
                close(row["actual_prefill_kv_cache_mb"], sum(p["adapter_prefill_kv_cache_mb"] for p in predictions) / 200)
                close(row["flops"], 650274495528.96)
        else:
            predictions = json.loads(source.with_name("predictions.json").read_text())
            total_key, prefill_key = "total_time_s", "prefilling_time_s"
            kv_key, flops_key = "kv_cache_mb", "prefill_flops"
            ref = next(r for r in rows if r["reference_group"] == row["reference_group"] and r["method"] == "base")
        assert len(predictions) == 200
        close(row["total_time_s"], sum(p[total_key] for p in predictions))
        close(row["prefilling_time_s"], sum(p[prefill_key] for p in predictions))
        close(row["kv_cache_mb"], sum(p[kv_key] for p in predictions) / 200)
        close(row["flops"], sum(p[flops_key] for p in predictions) / 200)
        close(row["reference_total_s"], ref["total_time_s"])
        close(row["reference_prefill_s"], ref["prefilling_time_s"])
        close(row["speedup_total"], ref["total_time_s"] / row["total_time_s"])
        close(row["speedup_prefilling"], ref["prefilling_time_s"] / row["prefilling_time_s"])
        if row["method"] in ("base", "qwen_base"):
            close(row["kv_cache_mb"], 47.448984375)
            close(row["flops"], 2660450546810.88)
        report.append({"method": row["method"], "group": row["reference_group"], "samples": 200,
                       "sum_of_sample_times_matches": True, "paired_speedups_match": True,
                       "mean_resources_match": True})
    output = {"status": "passed", "rows": report,
              "sample_set_matches_screenshot_resources": True,
              "historical_checkpoint_reproduced": False,
              "checkpoint_substitution": "current September Pixmo static KL step2000, explicitly authorized by user",
              "scope": "Arithmetic, sample set, cache accounting and saved protocol; does not prove identical historic hardware/software or stop policies across methods."}
    table.with_suffix(".validation.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
