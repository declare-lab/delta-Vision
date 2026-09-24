import unittest
from src.benchmarks import score_prediction, get_benchmark_spec
from src.benchmarks import anls, relaxed_correctness


class DocumentMetricsTests(unittest.TestCase):
    def test_chart_numeric_tolerance(self):
        self.assertEqual(relaxed_correctness('105', '100'), 1)
        self.assertEqual(relaxed_correctness('105.01', '100'), 0)
        self.assertEqual(relaxed_correctness('-95', '-100'), 1)
        self.assertEqual(relaxed_correctness('0.5', '50%'), 1)
        self.assertEqual(relaxed_correctness('0', '0'), 1)
        self.assertEqual(relaxed_correctness('1', '0'), 0)

    def test_no_generic_vqa_normalization(self):
        self.assertEqual(relaxed_correctness('The Company', 'the company'), 1)
        self.assertEqual(relaxed_correctness('company', 'the company'), 0)
        self.assertEqual(relaxed_correctness('1,000', '1000'), 0)

    def test_anls_reference_behavior(self):
        self.assertEqual(anls('HELLO   World', ['hello world']), 1)
        self.assertEqual(anls('abxx', ['abcd']), .5)
        self.assertEqual(anls('axxx', ['abcd']), 0)
        self.assertEqual(anls('abc', ['wrong', 'abc']), 1)
        self.assertEqual(anls('', ['answer']), 0)

    def test_registry_dispatch(self):
        for name in ['ChartQA', 'DocVQA', 'InfographicVQA']:
            spec = get_benchmark_spec(name)
            result = score_prediction(metric=spec.metric, prediction_text='hello', answer='HELLO')
            self.assertEqual(result['score'], 1)
        self.assertEqual(score_prediction(metric='chartqa_relaxed', prediction_text='105', answer='100')['score'], 1)


if __name__ == '__main__':
    unittest.main()
