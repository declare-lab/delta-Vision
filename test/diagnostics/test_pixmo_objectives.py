import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from analysis.table10_training_objective.pixmo_objective_comparison import FullVocabKL, trajectory_inputs, sample_trajectory


class Objectives(unittest.TestCase):
    def test_full_kl_and_gradient(self):
        torch.manual_seed(44)
        for temperature in (1.,2.):
            for chunk in (1,7,100):
                s=torch.randn(19,137,requires_grad=True)
                t=torch.randn_like(s)
                exact=F.kl_div(F.log_softmax(s/temperature,-1),F.softmax(t/temperature,-1),reduction='batchmean')*temperature**2
                grad=torch.autograd.grad(exact,s)[0]
                actual=FullVocabKL.apply(s,t,temperature,chunk)
                got=torch.autograd.grad(actual,s)[0]
                torch.testing.assert_close(actual,exact,atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(got,grad,atol=1e-7,rtol=1e-4)

    def test_token_trajectory_eos_and_padding(self):
        prompt=dict(input_ids=torch.tensor([[3,4,0],[3,4,5]]),attention_mask=torch.tensor([[1,1,0],[1,1,1]]),
                    mm_token_type_ids=torch.zeros(2,3,dtype=torch.long))
        tokens=[torch.tensor([6,7]),torch.tensor([2,8]),torch.tensor([0,2])]
        active=[torch.tensor([True,True]),torch.tensor([True,True]),torch.tensor([False,True])]
        inputs,mask=trajectory_inputs(prompt,tokens,active)
        self.assertEqual(inputs['input_ids'].tolist(),[[3,4,0,6,2,0],[3,4,5,7,8,2]])
        self.assertEqual(mask.sum(1).tolist(),[2,3])
        self.assertEqual(inputs['attention_mask'].tolist(),[[1,1,0,1,1,0],[1,1,1,1,1,1]])

    def test_student_sampling_not_gold_answers(self):
        prompt=dict(input_ids=torch.tensor([[3,4],[3,4]]),attention_mask=torch.ones(2,2,dtype=torch.long),
                    mm_token_type_ids=torch.zeros(2,2,dtype=torch.long))
        processor=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2,pad_token_id=0))
        logits=torch.zeros(2,1,10);mask=torch.ones(2,1,dtype=torch.bool)
        with patch('src.model.build_qwen_initial_context',return_value=(None,None)), \
             patch('src.model.qwen_embedding_adapter_prefill_cache',return_value=(logits,mask,{})), \
             patch('src.model.qwen_embedding_adapter_decode_step_shape_exact',return_value=(logits,{})) as decode, \
             patch('torch.multinomial',side_effect=[torch.tensor([[6],[7]]),torch.tensor([[2],[8]]),torch.tensor([[9],[2]])]):
            (inputs,answer),truncated=sample_trajectory(None,None,processor,prompt,3)
        self.assertEqual(inputs['input_ids'].tolist(),[[3,4,6,2,0],[3,4,7,8,2]])
        self.assertEqual(answer.sum(1).tolist(),[2,3])
        self.assertEqual(decode.call_count,2)
        self.assertEqual(truncated,0.)


if __name__=='__main__':unittest.main()
