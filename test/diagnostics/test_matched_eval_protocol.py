"""Prevent silent differences between pruning/base and adapter evaluation."""
import unittest
from src.multimodal_eval_inputs import benchmark_manifest

from src import adapter_nodeepstack_suite as adapter
from src import multimodal_baseline_suite as pruning


class MatchedProtocolTests(unittest.TestCase):
    def test_mmiu_uses_repaired_question(self):
        import json
        path = benchmark_manifest('mmiu')
        self.assertEqual(path.name, 'mmiu_context_and_question_v2.jsonl')
        with path.open() as file:
            rows = [json.loads(next(file)) for _ in range(1000)]
        self.assertTrue(all(row['source_question'].strip() in row['question'] for row in rows))

    def test_defaults_match(self):
        a = adapter.make_parser().parse_args([])
        b = pruning.make_parser().parse_args(['run', '--output', '/tmp/not-executed'])
        self.assertEqual(a.prompt_layout, b.prompt_layout)
        self.assertEqual(a.max_new_tokens, b.max_new_tokens)
        self.assertEqual(a.max_new_tokens, 128)

    def test_explicit_historical_protocol(self):
        options = ['--prompt-layout', 'interleaved', '--max-new-tokens', '8']
        a = adapter.make_parser().parse_args(options)
        b = pruning.make_parser().parse_args(['run', '--output', '/tmp/not-executed'] + options)
        self.assertEqual((a.prompt_layout, a.max_new_tokens), ('interleaved', 8))
        self.assertEqual((b.prompt_layout, b.max_new_tokens), ('interleaved', 8))

    def test_resume_rejects_ambiguous_or_mismatched_records(self):
        args = pruning.make_parser().parse_args(['run', '--output', '/tmp/not-executed'])
        for record in ({}, {'prompt_layout': 'interleaved', 'max_new_tokens': 128},
                       {'prompt_layout': 'media_first_v1', 'max_new_tokens': 8}):
            with self.assertRaises(ValueError):
                pruning.check_protocol(record, args)
        pruning.check_protocol({'prompt_layout': 'media_first_v1', 'max_new_tokens': 128}, args)


if __name__ == '__main__':
    unittest.main()
