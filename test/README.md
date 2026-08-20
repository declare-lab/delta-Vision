# Diagnostic Test Workspace

All diagnostic experiments live under this directory. Do not put diagnostic scripts,
temporary configs, logs, plots, or JSON/CSV outputs under `src/`, `scripts/`, or
`artifacts/`.

Directory layout:

```text
test/diagnostics/  Diagnostic experiment scripts.
test/configs/      Small config files for diagnostic runs.
test/results/      Local outputs, ignored by git.
```

Rules:

- Production training/eval code stays in `src/`.
- Stable user-facing wrappers stay in `scripts/`.
- Diagnostic experiments go in `test/diagnostics/`.
- Diagnostic outputs go in `test/results/`.
- If a diagnostic becomes part of the maintained pipeline, move only the reusable code into
  `src/` and keep the experiment runner in `test/diagnostics/`.

## Qwen Plateau Diagnostics

Use `test/diagnostics/qwen_embedding_adapter_plateau_diagnostics.py` to check whether the
Qwen `embedding_adapter` loss plateau is caused by unreachable teacher information or by the
learned adapter memory.

Example:

```bash
.venv/bin/python test/diagnostics/qwen_embedding_adapter_plateau_diagnostics.py \
  --model-path /lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct \
  --checkpoint artifacts/experiments/qwen_topk1024_freezeqkv/RUN_NAME/checkpoints/qwen_embedding_adapter_step500.pt \
  --data /lustre-data/leijingdi/code/delta-vision/artifacts/data_quality/pixmo_ama_full_valid.clean.jsonl \
  --image-root /lustre-data/leijingdi/code/delta-vision \
  --batch-size 1 \
  --oracle-modes adapter,recurrent_adapter,initial,teacher_layer_input,teacher_layer_output \
  --timing-runs 3 \
  --output-dir test/results/qwen_plateau_diagnostics
```

Outputs:

```text
diagnostics.json           Full payload.
oracle.csv                 Adapter vs oracle KL/top-1 agreement.
trajectory_by_layer.csv    Trajectory loss and hidden norm by layer.
adapter_visual_norms.csv   Adapter visual memory and delta norms by layer.
norms.csv                  Input text/visual norm summaries.
gates.csv                  Gate diagnostics when an adapter has gate parameters.
timing.csv                 Optional static source path vs test-only recurrent adapter timing.
per_sample.csv             Per-sample answer-token KL.
```

`recurrent_adapter` is test-only. It does not change `src/` or training/eval wrappers.
