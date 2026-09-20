import unittest
import torch
from src.qwen_native_order_norm import native_order_rmsnorm, native_order_norm_rope
from src.qwen_adapter_kernels import exact_rope


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeOrderNormTest(unittest.TestCase):
    def test_fused_norm_rope_rounding_and_strides(self):
        with torch.inference_mode():
            for b,n,h in [(1,1,8),(1,1,32),(1,130,32),(1,1008,8),(2,17,8)]:
                x = torch.randn(b,n,h,128,device='cuda',dtype=torch.bfloat16)
                w = torch.randn(128,device='cuda',dtype=torch.bfloat16)
                phase = torch.randn(b,n,64,device='cuda')
                embeddings = tuple(torch.cat([f(phase)]*2,-1).bfloat16() for f in [torch.cos,torch.sin])
                def reference():
                    xf = x.float()
                    normed = (xf * torch.rsqrt(xf.pow(2).mean(-1,keepdim=True)+1e-6)).bfloat16() * w
                    return exact_rope(normed.transpose(1,2), embeddings)
                def fused():
                    return native_order_norm_rope(x.transpose(1,2),w,1e-6,embeddings)
                self.assertTrue(torch.equal(reference(),fused()),(b,n,h))
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):fused()
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):out=fused()
                x.mul_(2)
                graph.replay()
                self.assertTrue(torch.equal(reference(),out))

    def test_matches_native_mean_order(self):
        torch.manual_seed(2751)
        with torch.inference_mode():
            for n in [128, 2560]:
                for rows in [1, 2, 3, 4, 7, 8, 15, 16, 17, 125, 130, 880, 1008, 4160, 8448]:
                    for scale in [0.0001, 1., 100.]:
                        x = (torch.randn(rows, n, device='cuda') * scale).bfloat16()
                        weight = torch.randn(n, device='cuda', dtype=torch.bfloat16)
                        y = x.float()
                        expected = (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True)+1e-6)).bfloat16() * weight
                        actual = native_order_rmsnorm(x, weight, 1e-6)
                        self.assertTrue(torch.equal(expected, actual), (n, rows, scale,
                            int(expected.ne(actual).sum()), float((expected-actual).abs().max())))


if __name__ == '__main__':unittest.main()
