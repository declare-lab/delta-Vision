import unittest
from types import SimpleNamespace
import torch
from src.qwen35_experiment import generate_evaluation_answer, score_evaluation_prediction


class EvaluationStopTests(unittest.TestCase):
    def test_benchmark_cap_overrides_global_limit(self):
        processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0, decode=lambda tokens, **kw: 'A'))
        config = {'evaluation_generation': {'max_new_tokens': 64, 'do_sample': False, 'unfinished_response': 'invalid_zero'}}
        for cap in (8, 16):
            with self.subTest(cap=cap):
                def generate(**kwargs):
                    self.assertEqual(kwargs['max_new_tokens'], cap)
                    return torch.tensor([[7, 8, 1, 99]])
                model = SimpleNamespace(generate=generate, generation_config=SimpleNamespace(eos_token_id=99))
                result = generate_evaluation_answer(model, processor, {'input_ids': torch.tensor([[7, 8]])},
                    {'answer': 'A', 'choices': ['red', 'blue']}, SimpleNamespace(metric='multi_choice'),
                    config, max_new_tokens=cap)
                self.assertEqual(result['max_new_tokens'], cap)
                self.assertEqual(result['score'], 1.)

    def test_eos_distinguished_from_equal_length_truncation(self):
        processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0, decode=lambda tokens, **kw: 'A'))
        config = {'evaluation_generation': {'max_new_tokens': 2, 'do_sample': False, 'unfinished_response': 'invalid_zero'}}
        row = {'answer': 'A', 'choices': ['red', 'blue']}
        for tail, expected in [([1,99],1.), ([1,2],0.)]:
            with self.subTest(tail=tail):
                def generate(**kwargs):
                    self.assertEqual(kwargs['max_new_tokens'],2)
                    self.assertFalse(kwargs['do_sample'])
                    self.assertTrue(kwargs['use_cache'])
                    return torch.tensor([[7,8]+tail])
                model=SimpleNamespace(generate=generate,generation_config=SimpleNamespace(eos_token_id=99))
                result=generate_evaluation_answer(model,processor,{'input_ids':torch.tensor([[7,8]])},
                                                   row,SimpleNamespace(metric='multi_choice'),config)
                self.assertEqual(result['score'],expected)
                self.assertEqual(result['stopped_by_eos'],bool(expected))
                self.assertEqual(result['hit_generation_limit'],not bool(expected))
                self.assertEqual(result['generated_token_ids'],tail)
                self.assertEqual(score_evaluation_prediction(result,row,'multi_choice')['score'],expected)


if __name__=='__main__':unittest.main()
