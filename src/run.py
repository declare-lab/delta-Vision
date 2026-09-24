"""One public entrypoint: python -m src.run {train,eval} --family FAMILY -- ...

No numerical/model imports occur until the protocol and family are resolved.
Existing parsers remain authoritative; unknown options fail instead of silently
changing a historical experiment's protocol.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('qwen', 'llava', 'qwen35')
MODEL_KEYS = {'model_path', 'dtype', 'attn_implementation', 'output_mode'}


def option_tokens(options):
    """Translate explicit config options, preserving order and false values."""
    if isinstance(options, list) and all(isinstance(x, str) for x in options):
        return options.copy()
    if not isinstance(options, dict):
        raise ValueError('Options must be an object or a list of CLI strings')
    tokens = []
    for key, value in options.items():
        if value is None:
            continue
        key = key.replace('_', '-')
        if not key or key.startswith('-'):
            raise ValueError(f'Invalid option name: {key!r}')
        if isinstance(value, bool):
            tokens.append('--' + ('' if value else 'no-') + key)
        elif isinstance(value, (str, int, float)):
            tokens.extend(['--' + key, str(value)])
        else:
            raise ValueError(f'Use CLI strings for non-scalar option {key}')
    return tokens


def resolve(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('train', 'eval'))
    parser.add_argument('--family', choices=FAMILIES)
    parser.add_argument('--config', type=Path, help='Shared model + train/eval JSON; paths are relative to cwd')
    parser.add_argument('--workflow', choices=('adapter', 'baseline'), default=None,
                        help='baseline evaluates a prepared/frozen six-baseline suite')
    parser.add_argument('--dry-run', action='store_true', help='Print routed options without loading a model')
    raw = list(sys.argv[1:] if argv is None else argv)
    # The explicit separator avoids consuming backend --help/--config options.
    split = raw.index('--') if '--' in raw else len(raw)
    args = parser.parse_args(raw[:split])
    forwarded = raw[split+1:] if split < len(raw) else []
    cfg = json.loads(args.config.read_text()) if args.config else {}
    if not isinstance(cfg, dict):
        parser.error('Config must be a JSON object')
    unknown = set(cfg) - {'family', 'model', 'train', 'eval', 'workflow'}
    if unknown:
        parser.error(f'Unknown config sections: {sorted(unknown)}')
    family = args.family or cfg.get('family')
    if family not in FAMILIES:
        parser.error('Specify --family or config.family')
    if args.family and cfg.get('family', family) != family:
        parser.error('--family conflicts with config.family')
    workflow = args.workflow or cfg.get('workflow', 'adapter')
    if workflow not in ('adapter', 'baseline'):
        parser.error('workflow must be adapter or baseline')
    if args.workflow and cfg.get('workflow', workflow) != workflow:
        parser.error('--workflow conflicts with config.workflow')
    common = cfg.get('model', {})
    if not isinstance(common, dict) or set(common) - MODEL_KEYS:
        parser.error(f'model supports only {sorted(MODEL_KEYS)}')
    if family == 'qwen35' and common:
        parser.error('Qwen3.5 uses the existing --run-dir/config.json for shared model/training settings')
    if workflow == 'baseline' and (args.task != 'eval' or common):
        parser.error('baseline is eval-only; model settings come from the prepared run, not model overrides')
    tokens = option_tokens(common) + option_tokens(cfg.get(args.task, {})) + forwarded
    if family != 'qwen35' and workflow == 'adapter':
        # The family is authoritative; reject conflicting legacy model-kind flags.
        for i, token in enumerate(tokens):
            kind = token.split('=', 1)[1] if token.startswith('--model-kind=') else (
                tokens[i+1] if token == '--model-kind' and i+1 < len(tokens) else None)
            if kind is not None and kind != family:
                parser.error('--model-kind conflicts with --family')
        tokens = ['--model-kind', family] + tokens
    return args.task, family, tokens, args.dry_run, workflow


def backend(task, family):
    """Resolve the same implementation for CLI and programmatic callers."""
    if task not in ('train', 'eval') or family not in FAMILIES:
        raise ValueError((task, family))
    if family == 'qwen35':
        dependencies = str(ROOT / 'artifacts/dependencies/qwen35_python')
        if dependencies not in sys.path:
            sys.path.insert(0, dependencies)
        from src.training import qwen35 as qwen35_worker
        return qwen35_worker
    if task == 'train':
        from src.training import engine as train
        return train
    from src import evaluate as eval_benchmarks
    return eval_benchmarks


def execute(task, family, tokens):
    engine = backend(task, family)
    argv = [task] + tokens if family == 'qwen35' else tokens
    args = engine.parse_args(argv)
    if family != 'qwen35' and args.model_kind != family:
        raise ValueError('Parsed model-kind conflicts with family')
    return engine.run(args)


def baseline_command(family, tokens):
    """Run the pinned worker in a fresh interpreter so src imports cannot mix.

The frozen suite config remains authoritative for prompts, methods, retention,
sample IDs and scoring. Newly prepared suites snapshot the shared model setup.
"""
    parser = argparse.ArgumentParser(description='Evaluate one prepared baseline-suite shard')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--model', required=True, help='Model key in the prepared config')
    parser.add_argument('--method', required=True, help='Method key in the prepared config')
    parser.add_argument('--suite', choices=('image', 'multimodal'), default='image')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(tokens)
    run = args.run.resolve()
    config = json.loads((run/'config.json').read_text())
    model = config['models'][args.model]
    kind = model.get('kind') or model.get('family')
    if kind != family:
        raise ValueError(f'Prepared model kind {kind!r} != requested family {family!r}')
    if args.method not in config['methods'] or not 0 <= args.shard < config['shards']:
        raise ValueError('Method or shard is outside the prepared protocol')
    jobs = json.loads((run/'jobs.json').read_text())
    matching = [j for j in jobs if j['model'] == args.model and j['method'] == args.method and
                j.get('suite', 'image') == args.suite]
    if not matching:
        raise ValueError('Requested job is not in this prepared suite')
    source = run/'source'
    # The original suite marks its image implementation as ready; remaining
    # jobs use the native-port worker. Both implementations can now coexist in
    # a source snapshot, so file existence alone must not choose the algorithm.
    if args.suite == 'image' and matching[0].get('ready') is True:
        current = ['image_worker.py', 'native_worker.py']
    else:
        current = ['native_worker.py', 'image_worker.py']
    # Preserve the prepared algorithm first, then resolve its source layout.
    # Existing frozen runs keep their original scoring implementation.
    candidates = [source/layout/name for name in current
                  for layout in ('baselines', 'evaluation/baselines')]
    candidates += [source/'scripts/repair_baseline_suite.py',
                   source/'scripts/rerun_baseline_suite.py']
    worker = next((p for p in candidates if p.is_file()), None)
    if worker is None:
        raise FileNotFoundError(f'No supported frozen baseline worker in {source}')
    if worker.name in ('image_worker.py', 'rerun_baseline_suite.py') and args.suite != 'image':
        raise ValueError('This frozen worker supports only the image suite')
    command = [sys.executable, str(worker), 'worker', '--run', str(run),
               '--model', args.model, '--method', args.method, '--shard', str(args.shard)]
    if worker.name in ('native_worker.py', 'repair_baseline_suite.py'):
        command.extend(['--suite', args.suite])
    if args.smoke:
        command.append('--smoke')
    return command, source


def main(argv=None):
    task, family, tokens, dry_run, workflow = resolve(argv)
    if dry_run:
        print(json.dumps(dict(task=task, family=family, workflow=workflow, arguments=tokens), indent=2))
        return
    if workflow == 'baseline':
        command, cwd = baseline_command(family, tokens)
        env = dict(os.environ, PYTHONPATH=str(ROOT/'artifacts/dependencies/qwen35_python')+':'+str(cwd),
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                   HF_HUB_OFFLINE='1', HF_HUB_DISABLE_PROGRESS_BARS='1',
                   PYTORCH_ALLOC_CONF='expandable_segments:True')
        subprocess.run(command, cwd=cwd, env=env, check=True)
        return
    return execute(task, family, tokens)


if __name__ == '__main__':
    main()
