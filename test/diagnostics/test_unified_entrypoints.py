"""Protocol routing must not silently override model family or legacy options."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.run import baseline_command, execute, main, option_tokens, resolve


class EntrypointTests(unittest.TestCase):
    def test_model_config_is_shared_and_tasks_stay_separate(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'run.json'
            p.write_text(json.dumps(dict(family='qwen',model=dict(model_path='/model',dtype='bfloat16'),
                train=dict(max_steps=2000,output_dir='/train'),
                eval=dict(benchmark='mmstar',max_new_tokens=8,output_dir='/eval'))))
            _,_,train,_,_ = resolve(['train','--config',str(p)])
            _,_,evaluation,_,_ = resolve(['eval','--config',str(p)])
            for args in (train,evaluation):
                self.assertEqual(args[args.index('--model-path')+1],'/model')
                self.assertEqual(args[args.index('--dtype')+1],'bfloat16')
            self.assertNotIn('--max-steps',evaluation)
            self.assertNotIn('--benchmark',train)

    def test_family_conflicts_fail(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            resolve(['train','--family','qwen','--','--model-kind=llava'])

    def test_help_reaches_backend(self):
        task,family,args,dry,_ = resolve(['eval','--family','qwen','--','--help'])
        self.assertEqual(args,['--model-kind','qwen','--help'])
        self.assertFalse(dry)

    def test_false_and_zero_are_not_dropped(self):
        self.assertEqual(option_tokens(dict(compile_adapter=False,max_samples=0)),
                         ['--no-compile-adapter','--max-samples','0'])

    def test_unknown_sections_fail(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run.json';p.write_text('{"family":"qwen","trian":{}}')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                resolve(['train','--config',str(p)])

    def test_backend_namespace_not_global_argv(self):
        with patch('src.run.backend') as factory:
            engine=factory.return_value
            engine.parse_args.return_value.model_kind='qwen'
            execute('train','qwen',['--max-steps','2'])
            engine.parse_args.assert_called_once_with(['--max-steps','2'])
            engine.run.assert_called_once_with(engine.parse_args.return_value)

    def test_qwen35_stage_and_method_preserved(self):
        with patch('src.run.backend') as factory:
            execute('eval','qwen35',['--run-dir','/run','--method','native','--shard','4'])
            factory.return_value.parse_args.assert_called_once_with(
                ['eval','--run-dir','/run','--method','native','--shard','4'])

    def test_baseline_cannot_enter_training(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            resolve(['train','--family','qwen','--workflow','baseline'])

    def test_public_main_dispatches_adapter(self):
        with patch('src.run.execute') as dispatch:
            main(['train','--family','qwen','--','--max-steps','2'])
            dispatch.assert_called_once_with('train','qwen',['--model-kind','qwen','--max-steps','2'])

    def test_baseline_isolated_frozen_worker_and_family_check(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);worker=root/'source/scripts/rerun_baseline_suite.py'
            worker.parent.mkdir(parents=True);worker.write_text('# frozen worker')
            (root/'config.json').write_text(json.dumps(dict(models={'qwen4b':{'kind':'qwen'}},methods=['dart'],shards=8)))
            (root/'jobs.json').write_text(json.dumps([dict(model='qwen4b',method='dart',suite='image')]))
            args=['--run',d,'--model','qwen4b','--method','dart','--shard','3']
            command,cwd=baseline_command('qwen',args)
            self.assertEqual(command[1],str(worker))
            self.assertEqual(cwd,root/'source')
            self.assertNotIn('--suite',command)
            self.assertEqual(command[-2:],['--shard','3'])
            with self.assertRaises(ValueError):baseline_command('llava',args)

    def test_baseline_new_frozen_layout_keeps_worker_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);worker=root/'source/baselines/native_worker.py'
            worker.parent.mkdir(parents=True);worker.write_text('# frozen native worker')
            (root/'config.json').write_text(json.dumps(dict(models={'qwen4b':{'kind':'qwen'}},methods=['fastv'],shards=8)))
            (root/'jobs.json').write_text(json.dumps([dict(model='qwen4b',method='fastv',suite='multimodal')]))
            command,cwd=baseline_command('qwen',['--run',d,'--model','qwen4b',
                '--method','fastv','--suite','multimodal','--shard','2','--smoke'])
            self.assertEqual(command[1],str(worker))
            self.assertEqual(cwd,root/'source')
            self.assertEqual(command[command.index('--suite')+1],'multimodal')
            self.assertEqual(command[-1],'--smoke')

    def test_previous_baseline_layout_and_mixed_snapshot_preserve_algorithm(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            old=root/'source/evaluation/baselines/image_worker.py'
            old.parent.mkdir(parents=True);old.write_text('# frozen image worker')
            (root/'config.json').write_text(json.dumps(dict(models={'qwen4b':{'kind':'qwen'}},methods=['dart'],shards=8)))
            (root/'jobs.json').write_text(json.dumps([dict(model='qwen4b',method='dart',suite='image',ready=True)]))
            args=['--run',d,'--model','qwen4b','--method','dart']
            command,_=baseline_command('qwen',args)
            self.assertEqual(command[1],str(old))
            new=root/'source/baselines/native_worker.py'
            new.parent.mkdir(parents=True);new.write_text('# frozen native worker')
            command,_=baseline_command('qwen',args)
            self.assertEqual(command[1],str(old))
            self.assertNotIn('--suite',command)

    def test_both_new_workers_preserve_the_prepared_algorithm(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'source/baselines'
            folder.mkdir(parents=True)
            for name in ['image_worker.py','native_worker.py']:(folder/name).write_text('# frozen')
            (root/'config.json').write_text(json.dumps(dict(models={'qwen4b':{'kind':'qwen'}},methods=['dart'],shards=8)))
            (root/'jobs.json').write_text(json.dumps([dict(model='qwen4b',method='dart',suite='image',ready=True)]))
            command,_=baseline_command('qwen',['--run',d,'--model','qwen4b','--method','dart'])
            self.assertEqual(command[1],str(folder/'image_worker.py'))
            self.assertNotIn('--suite',command)


if __name__=='__main__':unittest.main()
