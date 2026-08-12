"""Prepare mixed data by processing parquet shards one at a time (low memory)."""
import json, random, os
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
import io

output_dir = Path("data/mixed_v2")
output_dir.mkdir(parents=True, exist_ok=True)
img_dir = output_dir / "llava_images"
img_dir.mkdir(exist_ok=True)
output_path = output_dir / "train.jsonl"

# 1. Pixmo clean
print("Loading pixmo clean...")
pixmo_samples = []
pixmo_path = "/lustre-data/leijingdi/code/delta-vision/artifacts/data_quality/pixmo_ama_full_valid.clean.jsonl"
with open(pixmo_path) as f:
    for line in f:
        row = json.loads(line)
        pixmo_samples.append({
            "image": row["image"],
            "image_root": "/lustre-data/leijingdi/code/delta-vision",
            "question": row["question"],
            "answer": row["answer"],
            "source": "pixmo_clean",
        })
print(f"  Pixmo: {len(pixmo_samples)}")

# 2. LLaVA - process parquet shards one by one
print("Processing LLaVA parquet shards...")
parquet_dir = Path(os.path.expanduser("~/.cache/huggingface/hub/datasets--HuggingFaceM4--FineVision/snapshots/3c380a731a3429c1d04693d6ec16d7e683def84c/LLaVA_Instruct_150K/"))
shards = sorted(parquet_dir.glob("*.parquet"))
print(f"  Found {len(shards)} shards")

llava_samples = []
global_idx = 0
for shard_idx, shard_path in enumerate(shards):
    # Resolve symlink
    real_path = shard_path.resolve()
    table = pq.read_table(str(real_path), columns=["texts", "image"])
    for row_idx in range(len(table)):
        texts = table["texts"][row_idx].as_py()
        if not texts:
            global_idx += 1
            continue
        question = texts[0].get("user", "").strip()
        answer = texts[0].get("assistant", "").strip()
        if not question or not answer:
            global_idx += 1
            continue
        
        # Save image
        img_data = table["image"][row_idx].as_py()
        img_path = img_dir / f"{global_idx:06d}.jpg"
        if not img_path.exists():
            img = Image.open(io.BytesIO(img_data["bytes"]))
            img.save(img_path, "JPEG", quality=85)
        
        llava_samples.append({
            "image": str(img_path),
            "image_root": "",
            "question": question,
            "answer": answer,
            "source": "llava_instruct_150k",
        })
        global_idx += 1
    
    if (shard_idx + 1) % 10 == 0:
        print(f"  Shard {shard_idx+1}/{len(shards)}, total LLaVA: {len(llava_samples)}", flush=True)
    del table  # free memory

print(f"  LLaVA total: {len(llava_samples)}")

# 3. Combine and shuffle
all_samples = pixmo_samples + llava_samples
random.seed(42)
random.shuffle(all_samples)

with open(output_path, "w") as f:
    for s in all_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print(f"Done! {len(all_samples)} samples -> {output_path}")
