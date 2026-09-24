"""Check original-position causality, GQA, cached decode and graph replay."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.attention import prefix_plan, decode_plan, attention_heads


def main():
    torch.manual_seed(42)
    device = "cuda:0"
    errors = []
    with torch.inference_mode():
        for text, images in [([0, 1, 6, 7, 8], [2, 3, 4, 5]),
                             ([0, 4, 5, 9, 10], [1, 2, 3, 6, 7, 8]),
                             ([3, 4, 5], [0, 1, 2]), ([0, 1, 2], []), ([0, 1], [2, 3])]:
            tp = torch.tensor([text], device=device)
            ip = torch.tensor([images], device=device, dtype=torch.long)
            plan = prefix_plan(tp, ip, torch.ones_like(tp, dtype=torch.bool), torch.ones_like(ip, dtype=torch.bool))
            assert plan["dense_decode_ready"] == (not images or max(images) < text[-1])
            q = torch.randn(1, 8, len(text), 128, device=device, dtype=torch.bfloat16)
            k = torch.randn(1, 2, len(text) + len(images), 128, device=device, dtype=torch.bfloat16)
            v = torch.randn_like(k)
            positions = torch.cat([ip, tp], dim=1)
            mask = (positions[:, None, :] <= tp[:, :, None]).unsqueeze(1)
            expected = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                attn_mask=mask, enable_gqa=True).transpose(1, 2)
            actual = attention_heads(q, k, v, scaling=128 ** -.5, plan=plan)
            torch.testing.assert_close(actual.float(), expected, atol=.016, rtol=.016)
            errors.append(float((actual.float() - expected).abs().max()))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = attention_heads(q, k, v, scaling=128 ** -.5, plan=plan)
            graph.replay()
            assert torch.equal(actual, captured)
            # Alter a future visual key: the text before the image cannot change.
            if images and text[0] < images[-1]:
                changed = v.clone()
                changed[:, :, images.index(max(images))] += 100
                future = attention_heads(q, k, changed, scaling=128 ** -.5, plan=plan)
                assert torch.equal(actual[:, 0], future[:, 0])
            one_mask = mask[:, :, -1:]
            one_plan = decode_plan(one_mask)
            one = attention_heads(q[:, :, -1:], k, v, scaling=128 ** -.5, plan=one_plan)
            torch.testing.assert_close(one.float(), expected[:, -1:], atol=.016, rtol=.016)
    print({"cases": len(errors), "max_abs_errors_vs_fp32": errors, "graph_bitwise_equal": True})


if __name__ == "__main__":
    main()
