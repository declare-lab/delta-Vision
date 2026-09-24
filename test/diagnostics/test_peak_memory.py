import unittest
from unittest.mock import patch
from types import SimpleNamespace
from src.benchmarking.common.peak_memory import summarize_peak_memory


class PeakMemoryTest(unittest.TestCase):
    def test_maximum_of_each_request_not_sum_or_maximum_of_means(self):
        result = summarize_peak_memory([
            dict(prefill_peak_allocated_mb=100, decode_peak_allocated_mb=20),
            dict(prefill_peak_allocated_mb=30, decode_peak_allocated_mb=90)])
        self.assertEqual(result['peak_memory_mb'], 100)
        self.assertEqual(result['peak_memory_mb_mean'], 95)
        self.assertEqual(result['prefill_peak_memory_mb'], 100)
        self.assertEqual(result['decode_peak_memory_mb'], 90)

    def test_no_decode_preserves_continuation_peak_but_no_decode_stage(self):
        result = summarize_peak_memory([dict(adapter_prefill_peak_allocated_mb=80,
            adapter_decode_peak_allocated_mb=81, adapter_decode_steps=0)], prefix='adapter_')
        self.assertEqual(result['peak_memory_mb'], 81)
        self.assertIsNone(result['decode_peak_memory_mb'])

    def test_missing_measurement_is_not_zero(self):
        result = summarize_peak_memory([{}])
        self.assertIsNone(result['peak_memory_mb'])
        self.assertEqual(result['peak_memory_samples'], 0)

    def test_native_stage_boundary_and_request_maximum(self):
        import torch
        from src.benchmarking.common.generation_timing import GenerationStageTimer
        model = torch.nn.Linear(2, 2)
        timer = GenerationStageTimer(model)
        timer.device, timer.measure_memory = torch.device('cuda:0'), True
        layer = SimpleNamespace(keys=torch.zeros(1, 1, 2, 2), values=torch.zeros(1, 1, 2, 2),
            get_seq_length=lambda: 2)
        output = SimpleNamespace(past_key_values=SimpleNamespace(layers=[layer]))
        mib = 1024**2
        with patch('torch.cuda.synchronize'), patch('torch.cuda.reset_peak_memory_stats') as reset, \
             patch('torch.cuda.memory_allocated', return_value=80*mib), \
             patch('torch.cuda.max_memory_allocated', side_effect=[120*mib, 150*mib]), \
             patch('torch.cuda.max_memory_reserved', side_effect=[200*mib, 220*mib]):
            timer.begin()
            for count in [2, 1]:
                kwargs = dict(input_ids=torch.zeros(1, count, dtype=torch.long))
                timer._before(model, (), kwargs)
                timer._after(model, (), kwargs, output)
            result = timer.finish(1., 2)
        timer.remove()
        self.assertEqual(reset.call_count, 2)
        self.assertEqual(result['prefill_peak_allocated_mb'], 120)
        self.assertEqual(result['decode_peak_allocated_mb'], 150)
        self.assertEqual(result['peak_memory_mb'], 150)
        self.assertEqual(result['peak_memory_delta_mb'], 70)
        self.assertEqual(result['peak_reserved_mb'], 220)


if __name__ == '__main__':
    unittest.main()
