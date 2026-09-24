import unittest
import torch
from analysis.fig05_hybrid_attention.qwen35_memory_probe import spectrum_metrics, decompose, truncate, matrix_similarity, subspace_overlap, next_token_kl

class MemoryProbeTests(unittest.TestCase):
    def test_delta_rule_suffix_is_affine_in_boundary_state(self):
        torch.manual_seed(44)
        q,k,v=torch.randn(3,7,4,dtype=torch.float64)
        k=torch.nn.functional.normalize(k,dim=-1)
        beta=torch.rand(7,dtype=torch.float64)
        decay=torch.rand(7,dtype=torch.float64)
        def suffix(initial):
            state=initial.clone();out=[]
            for t in range(7):
                state=decay[t]*state
                error=v[t]-k[t]@state
                state=state+beta[t]*k[t,:,None]*error[None,:]
                out.append(q[t]@state)
            return torch.stack(out),state
        base,target,other=torch.randn(3,4,4,dtype=torch.float64)
        bo,bs=suffix(base);to,ts=suffix(target);oo,os=suffix(other)
        expected_o,expected_s=suffix(other+target-base)
        torch.testing.assert_close(oo+(to-bo),expected_o)
        torch.testing.assert_close(os+(ts-bs),expected_s)
    def test_energy_uses_squared_singular_values_and_heads_stay_separate(self):
        s=torch.tensor([[4.,1.,0.],[1.,1.,1.]])
        stats=spectrum_metrics(s,3)
        self.assertEqual(stats['r95_per_head'],[2,3])
        self.assertEqual(stats['r90_per_head'],[1,3])
    def test_zero_is_not_rank_one(self):
        stats=spectrum_metrics(torch.zeros(2,128),128)
        self.assertEqual(stats['r95_per_head'],[0,0])
        self.assertEqual(stats['effective_rank'],0.)
    def test_rank_zero_full_and_optimal_projection(self):
        x=torch.diag_embed(torch.tensor([[4.,2.,1.],[6.,3.,0.]]))
        svd=decompose(x)
        torch.testing.assert_close(truncate(svd,0),torch.zeros_like(x))
        torch.testing.assert_close(truncate(svd,3),x)
        torch.testing.assert_close(truncate(svd,1),torch.diag_embed(torch.tensor([[4.,0.,0.],[6.,0.,0.]])))
    def test_error_and_cosine_are_distinct(self):
        a=torch.tensor([[1.,2.]])
        stats=matrix_similarity(2*a,a)
        self.assertAlmostEqual(stats['cosine'],1.)
        self.assertAlmostEqual(stats['normalized_mse'],1.)
    def test_kl_uses_teacher_to_student_direction(self):
        teacher=torch.tensor([[2.,0.]])
        student=torch.tensor([[0.,0.]])
        expected=(teacher.softmax(-1)*(teacher.log_softmax(-1)-student.log_softmax(-1))).sum(-1)
        torch.testing.assert_close(next_token_kl(student,teacher),expected)
        self.assertEqual(next_token_kl(teacher,teacher).item(),0.)

if __name__=='__main__':unittest.main()
