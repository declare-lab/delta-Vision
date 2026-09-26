"""Check original-position causality, GQA, cached decode and graph replay."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.attention import prefix_plan, decode_plan, attention_heads


def check_batched_training():
    from src.model import qwen_prefix_causal_attention_mask
    device = 'cuda:0'
    # Different padding lengths, two visual spans, no visual tokens, and
    # text preceding/following images all share one training batch.
    texts = [[0, 1, 6, 7, 8], [0, 4, 5, 9, 10], [0, 1, 2], [3, 4]]
    images = [[2, 3, 4, 5], [1, 2, 3, 6, 7, 8], [], [0, 1, 2]]
    tp = torch.zeros(4, 5, device=device, dtype=torch.long)
    ip = torch.zeros(4, 6, device=device, dtype=torch.long)
    tm, im = torch.zeros_like(tp, dtype=torch.bool), torch.zeros_like(ip, dtype=torch.bool)
    for i, (text, image) in enumerate(zip(texts, images)):
        tp[i, :len(text)] = torch.tensor(text, device=device)
        ip[i, :len(image)] = torch.tensor(image, device=device)
        tm[i, :len(text)], im[i, :len(image)] = True, True
    plan = prefix_plan(tp, ip, tm, im)
    mask = qwen_prefix_causal_attention_mask(tm, im, torch.device(device), text_positions=tp, image_positions=ip)
    q = torch.randn(4, 8, 5, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(4, 2, 11, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    weight = torch.randn(4, 5, 8, 128, device=device) * tm[:, :, None, None]
    expected = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(),
        attn_mask=mask, enable_gqa=True).transpose(1, 2)
    expected_grads = torch.autograd.grad((expected * weight).sum(), (q, k, v))
    actual = attention_heads(q, k, v, scaling=128 ** -.5, plan=plan)
    actual_grads = torch.autograd.grad((actual.float() * weight).sum(), (q, k, v))
    torch.testing.assert_close(actual[tm].float(), expected[tm], atol=.016, rtol=.016)
    for grad, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(grad, reference, atol=.032, rtol=.032)
    # A loss in sample 0 must not backpropagate into another sample or padding.
    output = attention_heads(q, k, v, scaling=128 ** -.5, plan=plan)
    key_grad, value_grad = torch.autograd.grad(output[0].float().sum(), (k, v))
    assert torch.count_nonzero(key_grad[1:]) == torch.count_nonzero(value_grad[1:]) == 0
    assert torch.count_nonzero(key_grad[0, :, 4:6]) == torch.count_nonzero(value_grad[0, :, 4:6]) == 0
    print('Padded batch4 FA2 outputs, QKV gradients, padding and sample isolation: passed')


def main():
    torch.manual_seed(42)
    check_batched_training()
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
