"""Single-layer functional replacement, measured at answers rather than vectors.

Native prefix is reused exactly. At one layer, text queries read either adapter
visual K/V or no visual K/V. Image/prefix query outputs stay native. All later
layers run normally; the same intervention remains active during cached decode.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from src.adapter_single_layer_similarity import (
    ROOT, DATA, DATASETS, LAYERS, CHECKPOINT, CHECKPOINT_SHA, INPUT_KEYS,
    QwenBenchmarkDataset, _to_device_item, setup, sha, dump_json,
)
from src.benchmarks import get_benchmark_spec, score_prediction

PROTOCOL = "single_layer_visual_read_functional_v1"


class Experiment:
    def __init__(self):
        self.processor, self.model, self.adapter, self.cap, self.meta = setup()
        self.layers = self.model.model.language_model.layers
        self.kw = {}
        self.recording = False
        self.target, self.mode = None, "native"
        self.handles = []
        self.calls = 0
        for l in LAYERS:
            self.handles.append(self.layers[l].register_forward_pre_hook(self.save_kwargs(l), with_kwargs=True))
            self.handles.append(self.layers[l].self_attn.register_forward_pre_hook(self.attn_pre(l), with_kwargs=True))
            self.handles.append(self.layers[l].self_attn.register_forward_hook(self.attn_post(l), with_kwargs=True))

    def save_kwargs(self, l):
        def hook(module, args, kwargs):
            if self.recording:
                self.kw[l] = dict(kwargs)
        return hook

    def attn_pre(self, l):
        def hook(module, args, kwargs):
            if l != self.target:
                return
            self.calls += 1
            h = kwargs["hidden_states"]
            kw = dict(kwargs)
            if h.shape[1] == self.prompt_length and self.mode in ("adapter", "identity"):
                altered = h.clone()
                replacement = self.memories[l] if self.mode == "adapter" else self.cap.norm[l].index_select(1, self.visual)
                altered.index_copy_(1, self.visual, replacement)
                kw["hidden_states"] = altered
            elif self.mode == "no_read":
                cache = kwargs.get("past_key_values")
                length = h.shape[1] + (cache.get_seq_length(l) if cache is not None else 0)
                mask = torch.ones((1, length), device=h.device, dtype=torch.bool)
                mask[:, self.visual] = False
                kw["attention_mask"] = mask
            return args, kw
        return hook

    def attn_post(self, l):
        def hook(module, args, kwargs, output):
            if l != self.target or output[0].shape[1] != self.prompt_length:
                return
            # Preserve all image/prefix query outputs. Only text queries AFTER
            # the image are intervened. This isolates visual-to-text reading.
            out = output[0].clone()
            out[:, :self.query_start] = self.cap.out[l][:, :self.query_start]
            return (out, *output[1:])
        return hook

    def capture(self, inputs):
        self.inputs = inputs
        self.prompt_length = inputs["input_ids"].shape[1]
        types = inputs["mm_token_type_ids"][0]
        self.visual = types.eq(1).nonzero().flatten()
        assert len(self.visual) and types.max() == 1 and inputs["attention_mask"].bool().all()
        assert torch.equal(self.visual, torch.arange(self.visual[0], self.visual[-1] + 1, device=self.visual.device))
        self.query_start = int(self.visual[-1]) + 1
        assert self.query_start < self.prompt_length
        self.target, self.mode = None, "native"
        self.cap.enabled = self.recording = True
        self.cap.reset()
        self.model.model.rope_deltas = None
        self.positions = self.model._prepare_position_ids_for_generation(inputs["input_ids"], dict(inputs))
        out = self.model(**inputs, position_ids=self.positions, use_cache=True, logits_to_keep=1, return_dict=True)
        self.cap.check()
        self.cap.enabled = self.recording = False
        self.native_logits = out.logits[0, -1].float()
        self.native_cache = out.past_key_values
        initial = self.cap.h[0].index_select(1, self.visual)
        self.memories = {l: self.layers[l].input_layernorm(self.adapter.visual_memory_for_layer(initial, l)) for l in LAYERS}

    def copy_cache(self, count):
        cache = DynamicCache(config=self.model.config.text_config)
        for i in range(count):
            item = self.native_cache.layers[i]
            cache.update(item.keys.clone(), item.values.clone(), i)
        return cache

    def prefill(self, l=None, mode="native"):
        self.target, self.mode, self.calls = l, mode, 0
        if l is None:
            return self.native_logits.clone(), self.copy_cache(len(self.layers))
        cache = self.copy_cache(l)
        h = self.cap.h[l].clone()
        kw = dict(self.kw[l], past_key_values=cache)
        kw.pop("hidden_states", None)
        for j in range(l, len(self.layers)):
            h = self.layers[j](h, **kw)
        logits = self.model.lm_head(self.model.model.language_model.norm(h)[:, -1:])[0, -1].float()
        assert self.calls == 1, (l, mode, self.calls)
        return logits, cache

    def decode(self, token, cache, step):
        x = torch.tensor([[token]], device=self.visual.device, dtype=torch.long)
        positions = self.positions[:, :, -1:] + step
        out = self.model(input_ids=x, position_ids=positions, past_key_values=cache,
                         use_cache=True, logits_to_keep=1, return_dict=True)
        return out.logits[0, -1].float(), out.past_key_values

    def run(self, l=None, mode="native", forced=None, max_tokens=8):
        logits, cache = self.prefill(l, mode)
        tokens, distributions = [], []
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        steps = len(forced) if forced is not None else max_tokens
        for step in range(steps):
            distributions.append(logits.log_softmax(-1))
            token = int(forced[step]) if forced is not None else int(logits.argmax())
            tokens.append(token)
            if step + 1 == steps or (forced is None and token in eos):
                break
            logits, cache = self.decode(token, cache, step + 1)
        if l is not None:
            assert self.calls == len(tokens), (l, mode, self.calls, len(tokens))
        text = self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return dict(tokens=tokens, text=text), torch.stack(distributions)

    def close(self):
        self.cap.close()
        for h in self.handles:
            h.remove()


def score(name, item, prediction):
    return score_prediction(metric=get_benchmark_spec(name).metric, prediction_text=prediction["text"],
                            answer=item.get("answer"), answers=item.get("answers"), choices=item.get("choices"),
                            question=item["row"].get("question"))


def compare_distribution(native, replacement):
    assert native.shape == replacement.shape
    p = native.double().exp()
    kl = (p * (native.double() - replacement.double())).sum(-1)
    assert kl.min() > -1e-6, kl
    return dict(answer_token_kl=float(kl.mean().clamp_min(0)), first_token_kl=float(kl[0].clamp_min(0)),
                answer_token_tv=float((p - replacement.double().exp()).abs().sum(-1).mean() / 2),
                teacher_forced_steps=len(kl))


def prepare(root):
    plan = dict(protocol=PROTOCOL, model="Qwen3-VL-4B-Instruct", hidden_size=2560,
                checkpoint=str(CHECKPOINT), checkpoint_sha256=sha(CHECKPOINT), layers=list(LAYERS),
                layer_indexing="zero based", attention="flash_attention_2", deepstack="off", dtype="bfloat16",
                sources={str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), ROOT / "src/adapter_single_layer_similarity.py", ROOT / "src/model.py", ROOT / "src/benchmarks.py")},
                datasets={n: dict(path=str(DATA / f"{n}_eval.jsonl"), samples=c, sha256=sha(DATA / f"{n}_eval.jsonl")) for n,c in DATASETS.items()},
                conditions=["native", "each layer independently: adapter", "each layer independently: no_read"],
                intervention="one layer's text queries read adapter visual K/V; image/prefix query outputs remain native; native suffix; same visual cache during decode",
                no_read="at the same layer, exclude visual keys from text-query softmax, including decode; retain native image/prefix query outputs",
                generation="greedy EOS or benchmark max_new_tokens=8; no option-token stopping; no EOS suppression",
                kl="KL(native||replacement), full vocabulary at temperature 1, averaged along the SAME native greedy answer prefix including EOS; per-sample average then dataset average",
                agreement="normalized scored answers equal and both valid; two invalid answers do not count as agreement",
                limitation="single-layer robustness is not by itself evidence of retained visual function; interpret with same-layer no_read control")
    assert plan["checkpoint_sha256"] == CHECKPOINT_SHA
    path = root / "plan.json"
    if path.exists():
        assert json.loads(path.read_text()) == plan, "Output source/protocol mismatch"
    else:
        dump_json(path, plan)
    return plan


@torch.inference_mode()
def worker(args, smoke=False):
    root = Path(args.output)
    plan = prepare(root)
    exp = Experiment()
    begin = time.time()
    outpath = root / ("smoke_rows.jsonl" if smoke else f"rows_{args.shard}.jsonl")
    assert not outpath.exists(), outpath
    validation, completed = [], 0
    with outpath.open("w", buffering=1) as stream:
        for name, spec in plan["datasets"].items():
            ds = QwenBenchmarkDataset(spec["path"], exp.processor, name)
            assert len(ds) == spec["samples"]
            indices = [0] if smoke else list(range(args.shard, len(ds), args.world))
            for count, index in enumerate(indices):
                item = _to_device_item(ds[index], torch.device("cuda:0"))
                inputs = {k: v for k,v in item.items() if k in INPUT_KEYS}
                exp.capture(inputs)
                native, native_logp = exp.run(max_tokens=get_benchmark_spec(name).max_new_tokens)
                native["score"] = score(name, item, native)
                records = []
                identity_checks = {}
                for l in LAYERS:
                    if smoke:
                        identity, identity_logp = exp.run(l, "identity")
                        diff = float((identity_logp - native_logp).abs().max())
                        assert identity["tokens"] == native["tokens"] and diff == 0, (name,l,diff)
                        identity_checks[str(l)] = diff
                    for mode in ("adapter", "no_read"):
                        pred, free_logp = exp.run(l, mode, max_tokens=get_benchmark_spec(name).max_new_tokens)
                        pred["score"] = score(name, item, pred)
                        if pred["tokens"] == native["tokens"]:
                            aligned_logp = free_logp
                        else:
                            _, aligned_logp = exp.run(l, mode, forced=native["tokens"])
                        metrics = compare_distribution(native_logp, aligned_logp)
                        a, b = native["score"], pred["score"]
                        metrics.update(answer_agreement=bool(not a["invalid"] and not b["invalid"] and a["prediction"] == b["prediction"]),
                                       token_sequence_agreement=pred["tokens"] == native["tokens"])
                        records.append(dict(layer=l, mode=mode, prediction=pred, metrics=metrics))
                if smoke:
                    exp.target = None
                    exp.model.model.rope_deltas = None
                    hf = exp.model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=8)
                    hf_tokens = hf[0, inputs["input_ids"].shape[1]:].tolist()
                    assert native["tokens"] == hf_tokens, (native,hf_tokens)
                    validation.append(dict(dataset=name, native_generation_matches_HF=True, identity_max_logp_abs=identity_checks,
                                           visual_tokens=len(exp.visual), native_tokens=native["tokens"]))
                stream.write(json.dumps(dict(dataset=name, index=index, native=native, interventions=records), allow_nan=False) + "\n")
                completed += 1
                if count % 5 == 0 or count + 1 == len(indices):
                    print(json.dumps(dict(shard=args.shard, dataset=name, done=count+1, total=len(indices), elapsed=round(time.time()-begin,1))), flush=True)
    exp.close()
    dump_json(root / ("validation.json" if smoke else f"done_{args.shard}.json"),
              dict(complete=True, samples=completed, source_sha256=sha(__file__), plan_sha256=sha(root / "plan.json"),
                   seconds=time.time()-begin, checks=validation))


def summarize(native_scores, rows):
    scores = np.array([r["prediction"]["score"]["score"] for r in rows], dtype=float)
    delta = scores - np.array(native_scores)
    rng = np.random.default_rng(44)
    boot = rng.choice(delta, size=(4000, len(delta)), replace=True).mean(axis=1)
    return dict(samples=len(rows), accuracy=float(scores.mean()),
                accuracy_delta=float(delta.mean()), delta_paired_bootstrap_95ci=np.quantile(boot, [.025,.975]).tolist(),
                correct_to_wrong=int((delta < 0).sum()), wrong_to_correct=int((delta > 0).sum()),
                invalid=sum(r["prediction"]["score"]["invalid"] for r in rows),
                **{k:float(np.mean([r["metrics"][k] for r in rows])) for k in
                   ("answer_agreement", "token_sequence_agreement", "answer_token_kl", "first_token_kl", "answer_token_tv")})


def report(args):
    root = Path(args.output)
    plan = json.loads((root / "plan.json").read_text())
    all_rows = []
    for shard in range(args.world):
        done = json.loads((root / f"done_{shard}.json").read_text())
        assert done["complete"] and done["plan_sha256"] == sha(root / "plan.json")
        all_rows.extend(json.loads(x) for x in (root / f"rows_{shard}.jsonl").read_text().splitlines())
    keys = [(r["dataset"],r["index"]) for r in all_rows]
    expected = {(n,i) for n,s in plan["datasets"].items() for i in range(s["samples"])}
    assert len(keys) == len(set(keys)) and set(keys) == expected
    summary = {}
    text = ["# 单层功能替换：原生 vs embedding adapter", "",
            "Qwen3-VL-4B，现有 Pixmo KL 2000-step checkpoint；FA2、BF16、DeepStack 关闭。", "",
            "每个条件只改一层（0-based 13–22）。原生前缀不变，该层文本 query 读取 adapter 视觉 K/V；视觉/图像前缀 query 输出保持原生；后续层正常传播，decode 持续使用该层相同视觉 memory。", "",
            "no_read 在同一层阻断文本对视觉的读取，并保持其余视觉演化原生。单层替换无损若同时伴随 no_read 无损，不能归因于 adapter 保留视觉功能。", "",
            "准确率来自独立贪心生成，EOS 或最多 8 token 停止。答案一致率比较有效的归一化答案，两个无效答案不算一致。KL(native||replacement) 使用原生生成答案的相同前缀，全词表、温度 1，包含 EOS；先在答案 token 间平均再在样本间平均。", ""]
    for name, spec in plan["datasets"].items():
        rows = sorted((r for r in all_rows if r["dataset"] == name), key=lambda r:r["index"])
        native_scores = [r["native"]["score"]["score"] for r in rows]
        result = dict(samples=len(rows), native_accuracy=float(np.mean(native_scores)),
                      native_invalid=sum(r["native"]["score"]["invalid"] for r in rows), layers={})
        text.extend([f"## {name}（{len(rows)} 条），原生 acc={100*result['native_accuracy']:.2f}%", "",
                     "| 层 | Adapter acc (%) | Δacc (pp) | 答案一致率 (%) | 答案 KL | no_read acc (%) | no_read 一致率 (%) | no_read KL |",
                     "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for l in LAYERS:
            group = {}
            for mode in ("adapter", "no_read"):
                values = [next(x for x in r["interventions"] if x["layer"] == l and x["mode"] == mode) for r in rows]
                group[mode] = summarize(native_scores, values)
            sensitive = []
            for r in rows:
                by_mode = {x["mode"]:x for x in r["interventions"] if x["layer"] == l}
                if not r["native"]["score"]["invalid"] and not by_mode["no_read"]["metrics"]["answer_agreement"]:
                    sensitive.append(by_mode["adapter"]["metrics"]["answer_agreement"])
            group["no_read_changes_answer_samples"] = len(sensitive)
            group["adapter_preserves_native_among_no_read_changed"] = float(np.mean(sensitive)) if sensitive else None
            result["layers"][str(l)] = group
            a,n = group["adapter"],group["no_read"]
            text.append(f"| {l} | {a['accuracy']*100:.2f} | {a['accuracy_delta']*100:+.2f} | {a['answer_agreement']*100:.2f} | {a['answer_token_kl']:.5f} | {n['accuracy']*100:.2f} | {n['answer_agreement']*100:.2f} | {n['answer_token_kl']:.5f} |")
        text.append("")
        summary[name] = result
    text.extend(["配对 bootstrap Δacc 95% CI、正确→错误/错误→正确计数、无效回答数、逐样本输出和 no_read 敏感样本子集结果见 JSON。未预设等效性容忍界限，不将小样本内无显著差异解释为已证明功能等效。", ""])
    dump_json(root / "summary.json", summary)
    (root / "README.md").write_text("\n".join(text))
    dump_json(root / "status.json", dict(state="complete", samples=len(all_rows), single_layer_conditions=len(all_rows)*20))
    print("\n".join(text), flush=True)


def launch(args):
    root = Path(args.output)
    prepare(root)
    validation = json.loads((root / "validation.json").read_text())
    assert validation["complete"] and validation["source_sha256"] == sha(__file__)
    dump_json(root / "status.json", dict(state="running", pid=os.getpid(), gpus=list(range(args.world))))
    children, logs = [], []
    for shard in range(args.world):
        log = (root / f"worker_{shard}.log").open("w")
        logs.append(log)
        child = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-m", "src.adapter_single_layer_functional", "worker",
                                  "--output", str(root), "--world", str(args.world), "--shard", str(shard)], cwd=ROOT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false"),
                                 stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
    codes = [p.wait() for p in children]
    for log in logs:
        log.close()
    if any(codes):
        dump_json(root / "status.json", dict(state="failed", codes=codes))
        raise RuntimeError(codes)
    report(args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("smoke", "worker", "launch", "report"))
    p.add_argument("--output", default=str(ROOT / "artifacts/diagnostics/adapter_single_layer_functional_20260916"))
    p.add_argument("--world", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    args = p.parse_args()
    if args.mode == "smoke":
        worker(args, smoke=True)
    else:
        {"worker":worker, "launch":launch, "report":report}[args.mode](args)


if __name__ == "__main__":
    main()
