# Local baseline source snapshot

The main repository tracks `baselines/eval_baselines.py`,
`baselines/llava_hf_baselines.py`, `baselines/multimodal_pruning_utils.py`,
and the launch scripts explicitly. The six Qwen ports are preserved under
`baseline_overlays/ports/baselines/<method>/qwen3_vl/`, since their original
locations sit inside ignored independent Git repositories. Downloaded upstream
trees remain ignored.

`manifest.json` records the upstream URLs and exact local checkout commits.
The patches preserve additional edits to tracked upstream files against those
commits. On a fresh upstream checkout at the recorded commit, apply the
corresponding patch with `git -C baselines/<method> apply <absolute-patch-path>`.
Copy the contents of `dart_additions/` into `baselines/dart/`. Copy the
contents of `ports/baselines/` into the root `baselines/` directory after
preparing the upstream directories.
Do not reapply patches to the existing, already modified working trees.
Downloaded dependencies, weights, data, and run artifacts are excluded.

## Snapshot status

This is a working-code snapshot, not a claim that all evaluations have passed.
The new native baseline routes are selected by `scripts/repair_baseline_suite.py`;
legacy LLaVA routes in the older entry point are not all equivalent to these routes.
The historical MVBench input reconstruction still has an input-hash mismatch;
the evaluation guard rejects it rather than treating it as a matching rerun.
The rerun suites explicitly load the scorer from commit
`7f266415a28b3801339da93211a8fd9de2ff319e`; the current `src/benchmarks.py`
contains additional extraction changes and is not that frozen scorer.
