"""Bitwise regression of fused kernels against the previous BF16 FA2 path."""
import unittest
import torch
from src.qwen_adapter_kernels import exact_rope, split_attention_heads, pack_native_layer
from src.qwen_adapter_fa2 import attention_heads, prefix_plan_from_positions
from src.model import _apply_rope_one_from_embeddings


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ExactAdapterKernels(unittest.TestCase):
    def test_shared_video_prefix_is_exact_and_replays_new_kv(self):
        torch.manual_seed(8201)
        with torch.inference_mode():
            text=list(range(12));visual=[];start=12
            for _ in range(8):
                visual.extend(range(start,start+126));start+=126
                text.extend(range(start,start+5));start+=5
            text.extend(range(start,start+80))
            original_plan=prefix_plan_from_positions(text,visual,'cuda')
            shared_plan=prefix_plan_from_positions(text,visual,'cuda',shared_prefix=True)
            self.assertLess(shared_plan['key_indices'].numel(), original_plan['key_indices'].numel())
            query=torch.randn(1,32,len(text),128,device='cuda',dtype=torch.bfloat16)
            vk,vv=[torch.randn(1,len(visual),8,128,device='cuda',dtype=torch.bfloat16).transpose(1,2) for _ in range(2)]
            tk,tv=[torch.randn(1,len(text),8,128,device='cuda',dtype=torch.bfloat16).transpose(1,2) for _ in range(2)]
            def forward(plan):
                return split_attention_heads(query,vk,vv,tk,tv,scaling=128**-.5,plan=plan)
            self.assertTrue(torch.equal(forward(original_plan),forward(shared_plan)))
            gathered = attention_heads(query, torch.cat((vk,tk),2), torch.cat((vv,tv),2),
                scaling=128**-.5, plan=shared_plan)
            self.assertTrue(torch.equal(forward(original_plan),gathered))
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):forward(shared_plan)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):result=forward(shared_plan)
            vv.add_(3)
            graph.replay()
            self.assertTrue(torch.equal(result,forward(original_plan)))

    def test_packed_native_kv_layout_and_owned_replay(self):
        with torch.inference_mode():
            for visual_length, text_length in [(7, 3), (1008, 130)]:
                vk, vv = [torch.randn(1, visual_length, 8, 128, device='cuda', dtype=torch.bfloat16).transpose(1, 2) for _ in range(2)]
                tk, tv = [torch.randn(1, text_length, 8, 128, device='cuda', dtype=torch.bfloat16).transpose(1, 2) for _ in range(2)]
                output = torch.zeros(2, 3, 1, 8, visual_length + text_length, 128,
                    device='cuda', dtype=torch.bfloat16)
                def pack():
                    pack_native_layer(vk, vv, tk, tv, output[:, 1, 0])
                pack()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3): pack()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): pack()
                saved = output.clone()
                vk.add_(1)
                graph.replay()
                self.assertTrue(torch.equal(output[0, 1], torch.cat([vk, tk], dim=2)))
                self.assertTrue(torch.equal(output[1, 1], torch.cat([vv, tv], dim=2)))
                self.assertEqual(int(output[:, [0, 2]].count_nonzero()), 0)
                self.assertFalse(torch.equal(saved[0, 1], output[0, 1]))

    def test_rope_rounding_and_strides(self):
        torch.manual_seed(43)
        with torch.inference_mode():
            for batch, heads, length in [(1, 32, 1), (1, 8, 130), (1, 8, 1008), (2, 8, 17)]:
                x = torch.randn(batch, length, heads, 128, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
                phases = torch.randn(batch, length, 64, device='cuda')
                embeddings = tuple(torch.cat([fn(phases)] * 2, -1).to(torch.bfloat16) for fn in (torch.cos, torch.sin))
                expected = _apply_rope_one_from_embeddings(x, embeddings)
                actual = exact_rope(x, embeddings)
                self.assertTrue(torch.equal(expected, actual), (batch, heads, length))

    def test_split_attention_video_causality_and_replay(self):
        torch.manual_seed(44)
        with torch.inference_mode():
            for text, visual in [([0, 1, 9, 10], list(range(2, 9))),
                                 ([0, 5, 6, 15, 16], list(range(1, 5)) + list(range(7, 15))),
                                 ([4, 5, 6], list(range(4)))]:
                plan = prefix_plan_from_positions(text, visual, 'cuda')
                query = torch.randn(1, len(text), 32, 128, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
                vk, vv = [torch.randn(1, len(visual), 8, 128, device='cuda', dtype=torch.bfloat16).transpose(1, 2) for _ in range(2)]
                tk, tv = [torch.randn(1, len(text), 8, 128, device='cuda', dtype=torch.bfloat16).transpose(1, 2) for _ in range(2)]
                def reference():
                    return attention_heads(query, torch.cat([vk, tk], 2), torch.cat([vv, tv], 2), scaling=128**-.5, plan=plan)
                def fused():
                    return split_attention_heads(query, vk, vv, tk, tv, scaling=128**-.5, plan=plan)
                self.assertTrue(torch.equal(reference(), fused()))
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3): fused()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): output = fused()
                vv.add_(2)
                graph.replay()
                self.assertTrue(torch.equal(reference(), output))


if __name__ == '__main__':
    unittest.main()
