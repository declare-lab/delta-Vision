#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=${PY:-$ROOT_DIR/.venv/bin/python}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-235B-A22B-Instruct}
DEST=${DEST:-$ROOT_DIR/model/Qwen3-VL-235B-A22B-Instruct}

mkdir -p "$DEST"

echo "=== Download Qwen3-VL-235B ==="
echo "model_id=$MODEL_ID"
echo "dest=$DEST"

"$PY" - "$MODEL_ID" "$DEST" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

model_id = sys.argv[1]
dest = Path(sys.argv[2])
snapshot_download(
    repo_id=model_id,
    local_dir=str(dest),
    resume_download=True,
)
print(f"downloaded {model_id} to {dest}")
PY
