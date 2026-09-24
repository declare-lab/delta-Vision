import unittest
from src.data import sample_evaluation_rows


class SamplingTests(unittest.TestCase):
    def test_seed44_samples_whole_population_without_duplicates(self):
        import random
        rows = list(range(9000))
        selected, indices = sample_evaluation_rows(rows, seed=44)
        self.assertEqual(indices, sorted(random.Random(44).sample(range(9000), 1000)))
        self.assertEqual(selected, indices)
        self.assertEqual(len(set(indices)), 1000)
        self.assertNotEqual(indices, rows[:1000])
        self.assertEqual({index//3000 for index in indices}, {0,1,2})

    def test_small_dataset_uses_all_questions(self):
        selected, indices = sample_evaluation_rows(list(range(765)), seed=44)
        self.assertEqual(selected, list(range(765)))
        self.assertEqual(indices, selected)


if __name__=='__main__':
    unittest.main()
