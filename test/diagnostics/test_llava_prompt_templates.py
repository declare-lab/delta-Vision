"""Check real checkpoint tokenizers and teacher/student answer boundaries (CPU)."""
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from transformers import AutoProcessor

from src.data import VQADataset, LlavaBenchmarkDataset, llava_chat_text, llava_benchmark_prompt


ROOT = Path(__file__).resolve().parents[2]


class LlavaPromptTemplates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mistral = AutoProcessor.from_pretrained(ROOT/'model/llava-v1.6-mistral-7b-hf', local_files_only=True)
        cls.vicuna = AutoProcessor.from_pretrained(
            '/lustre-data/leijingdi/code/delta-vision/models/llava-1.5-7b-hf', local_files_only=True)

    def test_native_training_and_generation_text(self):
        question = 'What is shown?'
        self.assertEqual(llava_benchmark_prompt(self.mistral, question), '[INST] <image>\nWhat is shown? [/INST]')
        self.assertEqual(llava_chat_text(self.mistral, question, 'A cat.'),
                         '[INST] <image>\nWhat is shown? [/INST] A cat.</s> ')
        self.assertEqual(llava_chat_text(self.mistral, question, ''), llava_benchmark_prompt(self.mistral, question))

    def test_vicuna_training_compatibility(self):
        self.assertEqual(llava_benchmark_prompt(self.vicuna, 'Question'), 'USER: <image>\nQuestion\nASSISTANT:')
        self.assertEqual(llava_chat_text(self.vicuna, 'Question', 'Answer'),
                         'USER: <image>\nQuestion\nASSISTANT: Answer')

    def test_answer_boundaries_and_shared_image_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            Image.new('RGB', (96, 64), color='red').save(path/'image.png')
            rows = [dict(image='image.png',question=q,answer=a) for q,a in [
                ('What color is shown?', 'red'), ('图中是什么颜色？', '红色'), ('How many objects?', '2'),
            ]]
            (path/'data.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            for processor in [self.mistral, self.vicuna]:
                train = VQADataset(str(path/'data.jsonl'), processor)
                evaluate = LlavaBenchmarkDataset(str(path/'data.jsonl'), processor, 'gqa', answer_instruction='')
                for i in range(len(rows)):
                    with self.subTest(processor=type(processor).__name__,question=rows[i]['question']):
                        full, prompt = train[i], evaluate[i]
                        ids, prompt_ids = full['input_ids'], prompt['input_ids']
                        torch.testing.assert_close(ids[:len(prompt_ids)], prompt_ids, rtol=0, atol=0)
                        torch.testing.assert_close(full['pixel_values'], prompt['pixel_values'], rtol=0, atol=0)
                        image_id = int(processor.tokenizer.convert_tokens_to_ids(processor.image_token))
                        self.assertEqual(int((ids==image_id).sum()), int((prompt_ids==image_id).sum()))
                        visual_count = int((ids==image_id).sum())
                        self.assertEqual(full['prompt_len']+visual_count, len(prompt_ids))
                        text_ids = ids[ids!=image_id]
                        # The student and teacher start their loss on the same next-token target.
                        student_start = full['prompt_len']-1
                        teacher_start = visual_count+full['prompt_len']-1
                        self.assertEqual(int(text_ids[student_start+1]), int(ids[teacher_start+1]))
                        completion = processor.tokenizer.decode(ids[len(prompt_ids):],skip_special_tokens=True).strip()
                        self.assertEqual(completion, rows[i]['answer'])
                        if processor is self.mistral:
                            self.assertIn(processor.tokenizer.eos_token_id, ids[len(prompt_ids):].tolist())


if __name__ == '__main__':unittest.main()
