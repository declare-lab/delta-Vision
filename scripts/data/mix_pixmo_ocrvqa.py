"""Mix pixmo_clean + OCRvqa into training jsonl with source labels."""
import json
import random
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import io

output_dir = Path("data/pixmo_clean_ocrvqa")
output_dir.mkdir(parents=True, exist_ok=True)
img_dir = output_dir / "ocrvqa_images"
img_dir.mkdir(exist_ok=True)

samples = []

# 1. Pixmo clean (135K)
print("Loading pixmo clean...")
pixmo_path = "/lustre-data/leijingdi/code/delta-vision/artifacts/data_quality/pixmo_ama_full_valid.clean.jsonl"
with open(pixmo_path) as f:
    for line in f:
        row = json.loads(line)
        samples.append({
            "image": row["image"],
            "image_root": "/lustre-data/leijingdi/code/delta-vision",
            "question": row["question"],
            "answer": row["answer"],
            "source": "pixmo_clean",
        })
print(f"  pixmo_clean: {len(samples)}")

# 2. OCRvqa from FineVision (166K)
print("Loading OCRvqa...")
ds = load_dataset("HuggingFaceM4/FineVision", "ocrvqa", split="train")
print(f"  Total OCRvqa samples: {len(ds)}")

ocr_count = 0
for i, sample in enumerate(ds):
    texts = sample.get("texts", [])
    if not texts:
        continue
    question = texts[0].get("user", "").strip()
    answer = texts[0].get("assistant", "").strip()
    if not question or not answer:
        continue

    img = sample["images"][0] if isinstance(sample.get("images"), list) else sample["image"]
    img_path = img_dir / f"ocrvqa_{i:06d}.jpg"
    if not img_path.exists():
        img.convert("RGB").save(img_path, "JPEG", quality=85)

    samples.append({
        "image": str(img_path),
        "image_root": "",
        "question": question,
        "answer": answer,
        "source": "ocrvqa",
    })
    ocr_count += 1
    if (i + 1) % 20000 == 0:
        print(f"  OCRvqa: {i+1}/{len(ds)}...", flush=True)

print(f"  OCRvqa kept: {ocr_count}")

# Shuffle
random.seed(42)
random.shuffle(samples)

output_path = output_dir / "train.jsonl"
with open(output_path, "w") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nDone! Total: {len(samples)} samples")
print(f"  pixmo_clean: {len(samples) - ocr_count}")
print(f"  ocrvqa: {ocr_count}")
print(f"  Output: {output_path}")
