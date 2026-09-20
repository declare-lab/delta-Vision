import unittest
from src.embedding_adapter_corrected_eval import prompt_builder


class QuestionTests(unittest.TestCase):
    def setUp(self):self.build=prompt_builder()
    def test_options_only_context(self):
        doc={'context':'Candidates: A. No B. Yes','question':'Is the first image sharper?'}
        self.assertEqual(self.build(doc),'Candidates: A. No B. Yes\nQuestion: Is the first image sharper?')
    def test_generic_video_context(self):
        q='Why was the boy moving his arm?'
        self.assertIn(q,self.build({'context':'Examine 16 images. A: skating B: dancing','question':q}))
    def test_already_contains_question(self):
        self.assertEqual(self.build({'context':'What is this? A: apple','question':'What is this?'}),'What is this? A: apple')
    def test_no_context(self):self.assertEqual(self.build({'question':'Which image?'}),'Which image?')
    def test_no_question(self):
        with self.assertRaises(ValueError):self.build({'context':'A: yes B: no'})


if __name__=='__main__':unittest.main()
