"""Run the existing frozen-manifest evaluation with a bounded GPU allocation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import multimodal_baseline_suite as suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--step', type=int, default=2000)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--memory-gib', type=int, default=20)
    opts = parser.parse_args()
    train = opts.train_dir.resolve()
    checkpoint = train/f'checkpoints/qwen_embedding_adapter_step{opts.step}.pt'
    import torch
    torch.set_num_threads(4)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved.get('global_step', saved.get('step')) == opts.step
    assert saved['args']['output_mode'] == 'embedding_adapter'
    assert all(torch.isfinite(x).all() for x in saved['state_dict'].values())
    del saved
    reference = ROOT/'artifacts/experiments/qwen_mixed_adapter/qwen3vl4b_embedding_multi50_video50_rank128_4000_20260914_165124/post_train_eval'
    gpus = [int(x) for x in opts.gpus.split(',')]
    assert len(gpus) == len(set(gpus)) and gpus
    out = ROOT/'test/results'/f'{train.name}_step{opts.step}_{len(gpus)}gpu'
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out/'.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    suite.CHECKPOINT = checkpoint
    suite.ADAPTER_CHECKPOINTS['embedding_adapter'] = checkpoint
    original_popen = subprocess.Popen
    def capped_worker(command, **kwargs):
        assert command[1:4] == ['-m','src.multimodal_baseline_suite','worker']
        setup = ('import torch,runpy; '
                 f'torch.cuda.set_per_process_memory_fraction({opts.memory_gib}*1024**3/torch.cuda.get_device_properties(0).total_memory,0); '
                 'runpy.run_module("src.multimodal_baseline_suite",run_name="__main__")')
        return original_popen([command[0],'-c',setup,*command[3:]], **kwargs)
    results = {}
    suite.dump(out/'execution.json', dict(checkpoint=str(checkpoint), step=opts.step,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), gpus=gpus,
        torch_allocator_limit_gib=opts.memory_gib, training_not_paused=True,
        reference_manifests=str(reference/'manifest_map.json')))
    for bench, expected in [('muirbench',1000),('videomme',999),('mvbench',950)]:
        suite.dump(out/'status.json', dict(state='evaluating',benchmark=bench,results=results))
        args = suite.make_parser().parse_args(['run','--output',str(out/bench),
            '--methods','embedding_adapter','--checkpoint',str(checkpoint),'--benchmarks',bench,
            '--manifest-map',str(reference/'manifest_map.json'),'--limit','1000',
            '--shards',str(len(gpus)),'--gpus',*[str(gpu) for gpu in gpus],'--max-new-tokens','8' if bench=='muirbench' else '128',
            '--prompt-layout','media_first_v1','--json-only'])
        subprocess.Popen = capped_worker
        try:
            suite.run(args)
        finally:
            subprocess.Popen = original_popen
        rows = [json.loads(line) for f in (out/bench).glob('embedding_adapter_shard*.jsonl') for line in f.open()]
        old = {r['index']:r for f in (reference/bench).glob('embedding_adapter_shard*.jsonl')
               for line in f.open() for r in [json.loads(line)]}
        assert len(rows)==len(old)==expected
        for row in rows:
            prior = old[row['index']]
            for key in ('input_sha256','dataset_sha256','max_new_tokens','prompt_layout','deepstack_enabled'):
                assert row[key]==prior[key], (bench,row['index'],key)
        results[bench] = json.loads((out/bench/'results.json').read_text())[0]
        suite.dump(out/'results.json',results)
        print('COMPLETE',bench,results[bench],flush=True)
    suite.dump(out/'status.json',dict(state='complete',results=results))
    print('ALL COMPLETE',out,flush=True)


if __name__=='__main__':
    main()
