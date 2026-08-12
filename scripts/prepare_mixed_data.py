"""Merge pixmo clean + LLaVA-665K into unified {image, question, answer} format."""
import json
from pathlib import Path

output_path = "data/mixed_train.jsonl"
count = 0

with open(output_path, "w") as out:
    # 1. Pixmo clean (already in our format)
    pixmo_path = "/lustre-data/leijingdi/code/delta-vision/artifacts/data_quality/pixmo_ama_full_valid.clean.jsonl"
    with open(pixmo_path) as f:
        for line in f:
            row = json.loads(line)
            # image path is relative to delta-vision root
            out.write(json.dumps({
                "image": row["image"],
                "question": row["question"],
                "answer": row["answer"],
            }, ensure_ascii=False) + "\n")
            count += 1
    print(f"Pixmo clean: {count} samples")

    # 2. LLaVA-665K (conversations format -> question/answer)
    llava_path = "data/llava-665k/llava_v1_5_mix665k.json"
    llava_root = "data/llava-665k/train_split"
    with open(llava_path) as f:
        data = json.load(f)
    llava_count = 0
    for item in data:
        if "image" not in item:
            continue
        convs = item["conversations"]
        # Take first human/gpt pair
        if len(convs) >= 2 and convs[0]["from"] == "human" and convs[1]["from"] == "gpt":
            question = convs[0]["value"].replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "").strip()
            answer = convs[1]["value"].strip()
            image_path = f"{llava_root}/{item["image"]}"
            out.write(json.dumps({
                "image": image_path,
                "question": question,
                "answer": answer,
            }, ensure_ascii=False) + "\n")
            llava_count += 1
            count += 1
    print(f"LLaVA-665K: {llava_count} samples")

print(f"Total mixed: {count} samples -> {output_path}")
