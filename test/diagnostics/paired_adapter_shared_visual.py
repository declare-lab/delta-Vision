"""Try sharing immutable visual input buffers; keep the paired base unchanged."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).parent))
from functools import partial
import paired_runtime_execution

if __name__ == "__main__":
    paired_runtime_execution.build_qwen_fast_adapter_prefill = partial(
        paired_runtime_execution.build_qwen_fast_adapter_prefill, native_decode=False)
    paired_runtime_execution.main()
