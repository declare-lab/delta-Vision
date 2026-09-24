"""Export completed native-cache accuracy results as CSV and a rank curve."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    payload = json.loads((args.root / "summary.json").read_text())
    from transformers import AutoConfig
    plan = json.loads((args.root / "plan.json").read_text())
    dimensions = {name: AutoConfig.from_pretrained(path).text_config.hidden_size
                  for name, path in plan["models"].items()}
    payload["model_hidden_dimensions"] = dimensions
    (args.root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    conditions = ["native", "r0", "r32", "r64", "r128", "r256", "r512", "r1024"]
    table = ["| Model | Original hidden D | Dataset | N | Native | r0 | r32 | r64 | r128 | r256 | r512 | r1024 |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    with (args.root / "accuracy.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "original_hidden_dim", "benchmark", "samples", *conditions])
        for key, result in payload["results"].items():
            model, benchmark = key.split("/")
            writer.writerow([model, dimensions[model], benchmark, result["native"]["samples"],
                             *(round(result[c]["accuracy_pct"], 4) for c in conditions)])
            table.append("| " + " | ".join([model, str(dimensions[model]), benchmark,
                         str(result["native"]["samples"]),
                         *(f"{result[c]['accuracy_pct']:.2f}" for c in conditions)]) + " |")
    report = args.root / "RESULTS.md"
    lines = report.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Model |"))
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    report.write_text("\n".join([*lines[:start], *table, *lines[end:]]) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ranks = [32, 64, 128, 256, 512, 1024]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, model, title in zip(axes, ["qwen", "llava"], ["Qwen3-VL-4B", "LLaVA-1.5-7B"]):
        for benchmark, label, color in [("sqa", "ScienceQA", "#2563eb"),
                                         ("realworldqa", "RealWorldQA", "#dc2626"),
                                         ("mmstar", "MMStar", "#059669")]:
            results = payload["results"][f"{model}/{benchmark}"]
            delta = [results[f"r{r}"]["delta_accuracy_pp"] for r in ranks]
            ci = [results[f"r{r}"]["paired_bootstrap_delta_95ci_pp"] for r in ranks]
            ax.plot(ranks, delta, marker="o", ms=4, lw=1.5, color=color, label=label)
            ax.fill_between(ranks, [x[0] for x in ci], [x[1] for x in ci], color=color, alpha=.12)
        ax.axhline(0, color="black", lw=.8, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ranks, [str(r) for r in ranks])
        ax.set_xlabel("Channel bottleneck width r")
        ax.set_title(f"{title} (D = {dimensions[model]})")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Accuracy change from native (percentage points)")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("All visual tokens retained; native visual cache restored at every layer", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.root / "accuracy_vs_rank.pdf", bbox_inches="tight")
    fig.savefig(args.root / "accuracy_vs_rank.png", dpi=200, bbox_inches="tight")
    print(args.root / "accuracy.csv")
    print(args.root / "accuracy_vs_rank.pdf")


if __name__ == "__main__":
    main()
