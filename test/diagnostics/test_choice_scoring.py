"""Regression cases for prose/unfinished-reasoning falsely scored as a choice."""
import unittest
from src.benchmarks import canonical_choice, extract_choice, score_prediction, score_realworldqa_prediction


class ChoiceScoringTests(unittest.TestCase):
    def test_real_truncated_outputs_are_not_answers(self):
        for text in ['We are given a 2x3', 'We are given a stem-and-leaf',
                     'The image shows a grid with three visible',
                     'A cat is sitting', 'The image shows two samples, A and']:
            with self.subTest(text=text):
                self.assertIsNone(extract_choice(text, ['x','y','z','w']))
                self.assertEqual(score_prediction(metric='multi_choice', prediction_text=text,
                                                  answer='A', choices=['x','y','z','w'])['score'], 0)

    def test_explicit_answers(self):
        for text, expected in [('A','A'),('b','B'),('(C)','C'),('D. Explanation','D'),
                               ('**B**','B'),('The answer is C.','C'),
                               ('正确选项是：**A**\n\n解析','A'),
                               ('Answer is A. Final answer: B.','B'),
                               ('<think>Option A</think> D','D'),
                               ('The two samples have different temperatures.\n\nA. sample A','A')]:
            with self.subTest(text=text):
                self.assertEqual(extract_choice(text,['x','y','z','w']),expected)
        self.assertIsNone(extract_choice('The answer is a cat',['x','y','z','w']))
        self.assertIsNone(extract_choice('Option A is incorrect; I need to examine the others.', ['x','y','z','w']))

    def test_option_text_requires_whole_match(self):
        choices=['red','blue','green','yellow']
        self.assertEqual(extract_choice('blue',choices),'B')
        self.assertIsNone(extract_choice('It is not red, perhaps blue',choices))
        self.assertIsNone(extract_choice('red',['red','red']))
        self.assertEqual(canonical_choice('blue',choices),'B')
        self.assertEqual(canonical_choice('green',choices),'C')
        self.assertEqual(canonical_choice('C.',choices),'C')

    def test_final_answer_overrides_earlier_answer(self):
        self.assertEqual(extract_choice('The answer is A.\nAfter checking again:\nB', ['x','y','z','w']), 'B')
        self.assertEqual(score_realworldqa_prediction('A. first\nFinal answer: B', 'B', ['first','second'])['score'], 1)
        self.assertIsNone(extract_choice('A. red\nB. blue', ['red','blue']))
        self.assertEqual(score_prediction(metric='pope_f1', prediction_text='Yes. Correction: no.', answer='no')['score'], 1)

    def test_realworld_negation_and_conflicting_counts(self):
        self.assertEqual(score_realworldqa_prediction('There are not 3 cars; there are 4.', '3')['score'], 0)
        self.assertEqual(score_realworldqa_prediction('It is not blue.', 'blue')['score'], 0)
        self.assertEqual(score_realworldqa_prediction('There are 3 cars.', '3')['score'], 1)
        self.assertEqual(score_realworldqa_prediction('Initially I counted 3. Final answer: 4', '4')['score'], 1)


if __name__=='__main__':
    unittest.main()
