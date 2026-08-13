"""Benchmark prefill speed: teacher vs adapter, with torch.compile."""
import argparse, json, sys, time
sys.path.insert(0, ".")
import torch
from PIL import Image

def benchmark(fn, warmup=5, n_runs=100):
    for _ in range(warmup):
        with torch.no_grad():
            fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs


def run_qwen3vl(args):
    from src.model import (
        QwenVisualDeltaAdapter,
        build_qwen_initial_context,
        load_qwen_visual_delta_checkpoint,
        qwen_visual_delta_logits,
    )
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    device = "cuda:0"
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with open(args.sample_jsonl) as f:
        row = json.loads(f.readline())
    img = Image.open(args.data_root + "/" + row["image"]).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": row["question"]}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True).to(device)
    mm_ids = inputs.get("mm_token_type_ids")

    adapter = None
    if args.checkpoint:
        adapter, _ = load_qwen_visual_delta_checkpoint(args.checkpoint, model.model.language_model, torch.device(device), torch.bfloat16)
    else:
        adapter = QwenVisualDeltaAdapter.from_language_model(
            model.model.language_model,
            mode="native_visual_kv_split",
        ).to(device=device, dtype=torch.bfloat16)
        adapter.eval()

    with torch.no_grad():
        initial_hidden, position_ids = build_qwen_initial_context(model, dict(inputs))

    # Compile
    if args.compile:
        compiled_teacher = torch.compile(lambda: model(input_ids=inputs.input_ids, pixel_values=inputs.pixel_values, image_grid_thw=inputs.image_grid_thw, mm_token_type_ids=mm_ids).logits, mode="max-autotune")
        compiled_adapter = torch.compile(lambda: qwen_visual_delta_logits(model, adapter, dict(inputs), initial_hidden=initial_hidden, position_ids=position_ids)[0], mode="max-autotune")
        for _ in range(3):
            with torch.no_grad():
                compiled_teacher()
                compiled_adapter()
        torch.cuda.synchronize()

    # Benchmark
    t_teacher = benchmark(lambda: model(input_ids=inputs.input_ids, pixel_values=inputs.pixel_values, image_grid_thw=inputs.image_grid_thw, mm_token_type_ids=mm_ids).logits, n_runs=args.n_runs)
    t_e2e = benchmark(lambda: qwen_visual_delta_logits(model, adapter, dict(inputs))[0], n_runs=args.n_runs)
    t_cached = benchmark(lambda: qwen_visual_delta_logits(model, adapter, dict(inputs), initial_hidden=initial_hidden, position_ids=position_ids)[0], n_runs=args.n_runs)

    n_vis = int((inputs.mm_token_type_ids == 1).sum().item())
    n_text = int((inputs.mm_token_type_ids == 0).sum().item())
    print(f"\n=== {args.model_path} Prefill ({args.n_runs} runs) ===")
    print(f"Visual tokens: {n_vis}, Text tokens: {n_text}")
    print(f"Teacher:          {t_teacher*1000:6.1f} ms  (1.00x)")
    print(f"Adapter e2e:      {t_e2e*1000:6.1f} ms  ({t_teacher/t_e2e:.2f}x)")
    print(f"Adapter V0 cached:{t_cached*1000:6.1f} ms  ({t_teacher/t_cached:.2f}x)")




def run_llava(args):
    from src.model import PerLayerKVAdapter, extract_vision_kv, load_frozen_llava, student_forward_with_visual_kv, load_adapter_checkpoint

    device = "cuda:0"
    processor, model = load_frozen_llava(args.model_path, dtype=torch.bfloat16, device=device)
    image_token_id = int(getattr(model.config, "image_token_index", 32000))

    with open(args.sample_jsonl) as f:
        row = json.loads(f.readline())
    img = Image.open(args.data_root + "/" + row["image"]).convert("RGB")
    prompt = "USER: <image>\n" + row["question"] + "\nASSISTANT:"
    prompt = "USER: <image>\n" + row["question"] + "\nASSISTANT:"
    prompt = "USER: <image>\n" + row["question"] + "\nASSISTANT:"
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)

    adapter = None
    if args.checkpoint:
        adapter, _, _ = load_adapter_checkpoint(args.checkpoint, device, language_model=model.model.language_model)

    with torch.no_grad():
        source_k, source_v = extract_vision_kv(model, inputs.pixel_values, [22, 23])

    # Compile
    if args.compile:
        compiled_teacher = torch.compile(lambda: model(input_ids=inputs.input_ids, pixel_values=inputs.pixel_values).logits, mode="max-autotune")
        compiled_vit = torch.compile(lambda: extract_vision_kv(model, inputs.pixel_values, [22, 23]), mode="max-autotune")
        compiled_student = torch.compile(lambda sk, sv: student_forward_with_visual_kv(model, inputs.input_ids, adapter, sk, sv, image_token_id), mode="max-autotune")
        for _ in range(3):
            with torch.no_grad():
                compiled_teacher()
                sk, sv = compiled_vit()
                compiled_student(sk.clone(), sv.clone())
        torch.cuda.synchronize()

    t_teacher = benchmark(lambda: model(input_ids=inputs.input_ids, pixel_values=inputs.pixel_values).logits, n_runs=args.n_runs)
    t_e2e = benchmark(lambda: (lambda sk_, sv_: student_forward_with_visual_kv(model, inputs.input_ids, adapter, sk_, sv_, image_token_id))(*extract_vision_kv(model, inputs.pixel_values, [22, 23])), n_runs=args.n_runs)
    t_cached = benchmark(lambda: student_forward_with_visual_kv(model, inputs.input_ids, adapter, source_k, source_v, image_token_id), n_runs=args.n_runs)

    n_vis = source_k.shape[2]
    n_text = (inputs.input_ids[0] != image_token_id).sum().item()
    print()
    print(f"=== {args.model_path} Prefill ({args.n_runs} runs) ===")
    print(f"Visual tokens: {n_vis}, Text tokens: {n_text}")
    print(f"Teacher:          {t_teacher*1000:6.1f} ms  (1.00x)")
    print(f"Ours e2e:         {t_e2e*1000:6.1f} ms  ({t_teacher/t_e2e:.2f}x)")
    print(f"Ours KV cached:   {t_cached*1000:6.1f} ms  ({t_teacher/t_cached:.2f}x)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", default=None, help="Adapter checkpoint (optional, random init if not set)")
    parser.add_argument("--sample-jsonl", default="../delta-vision/data/mmstar/mmstar_val.jsonl")
    parser.add_argument("--data-root", default="../delta-vision")
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--compile", action="store_true", default=True)
    args = parser.parse_args()

    if "qwen" in args.model_path.lower():
        run_qwen3vl(args)
    elif "llava" in args.model_path.lower():
        run_llava(args)
    else:
        print("Supported: qwen3-vl, llava models")


if __name__ == "__main__":
    main()
