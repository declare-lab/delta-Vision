"""Locate adapter/native graph overhead; instrumentation is not a speed table."""
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import torch
from src.graphs import AdapterDecodeGraph
from src.graphs import NativeDecodeGraph
import paired_runtime_execution

rows = []


def instrument(cls, label):
    original = cls.replay

    def replay(self, *args, **kwargs):
        graph_replay = self.graph.replay
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        def observed():
            begin.record()
            graph_replay()
            end.record()

        self.graph.replay = observed
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            output = original(self, *args, **kwargs)
            launch_done = time.perf_counter()
            torch.cuda.synchronize()
            rows.append(dict(method=label, wall_ms=1000*(time.perf_counter()-start),
                cpu_launch_ms=1000*(launch_done-start), graph_gpu_ms=begin.elapsed_time(end)))
            return output
        finally:
            self.graph.replay = graph_replay

    cls.replay = replay


if __name__ == "__main__":
    instrument(AdapterDecodeGraph, "embedding_adapter")
    instrument(NativeDecodeGraph, "base")
    paired_runtime_execution.main()
    result = dict(scope="Instrumented graph replay including input/output copies; CUDA events delimit graph kernels. Not a benchmark speedup.",
        rows=rows, medians={label: {k: statistics.median(r[k] for r in rows if r['method']==label)
            for k in ['wall_ms','cpu_launch_ms','graph_gpu_ms']} for label in ['base','embedding_adapter']})
    (ROOT/'test/results/runtime_optimized_20260915/adapter_decode_graph_cost.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['medians']), flush=True)
