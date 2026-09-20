from __future__ import annotations

import argparse, json, os, subprocess, sys, time, random
from pathlib import Path
from typing import Any

import torch

from src.benchmarks import get_benchmark_spec, score_prediction
from src.data import QwenBenchmarkDataset, LlavaBenchmarkDataset
from src.eval_benchmarks import extract_option_from_text
from src.model import load_frozen_qwen3vl, load_frozen_llava

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QWEN = "/lustre-data/leijingdi/code/delta-vision/models/Qwen3-VL-4B-Instruct"
DEFAULT_LLAVA = str(ROOT / "models/llava-1.5-7b-hf")


def _to_device_item(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in item.items():
        out[k] = v.to(device).unsqueeze(0) if torch.is_tensor(v) and v.ndim > 0 and k not in {"pixel_values", "image_grid_thw"} else v
    # dataset tensors differ by model
    if "pixel_values" in item and torch.is_tensor(item["pixel_values"]):
        pv = item["pixel_values"].to(device)
        if pv.ndim == 3:
            pv = pv.unsqueeze(0)
        out["pixel_values"] = pv
    if "image_grid_thw" in item and torch.is_tensor(item["image_grid_thw"]):
        out["image_grid_thw"] = item["image_grid_thw"].to(device)
    if "image_sizes" in item and torch.is_tensor(item["image_sizes"]):
        im = item["image_sizes"].to(device)
        if im.ndim == 1:
            im = im.unsqueeze(0)
        out["image_sizes"] = im
    return out


def _dataset(model_kind: str, processor, benchmark: str, samples: int | None, data: str | None = None):
    spec = get_benchmark_spec(benchmark)
    path = ROOT / spec.default_data if data is None else Path(data)
    cls = QwenBenchmarkDataset if model_kind == "qwen" else LlavaBenchmarkDataset
    return cls(str(path), processor, benchmark, data_root=str(path.parent), max_samples=samples)


def _layers(model_kind: str, model):
    if model_kind == "qwen":
        return model.model.language_model.layers
    # Llava hf variants
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        return lm.model.layers if hasattr(lm, "model") else lm.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Could not locate LLaVA language layers")


class VisualChannelHook:
    def __init__(self, model_kind: str, model, *, mode: str, rank: int | None = None, basis: dict | None = None,
                 max_rows: int = 2048, seed: int = 44):
        self.model_kind = model_kind
        self.model = model
        self.mode = mode
        self.rank = rank
        self.basis = basis or {}
        self.max_rows = int(max_rows)
        self.rng = random.Random(seed)
        self.handles = []
        self.rows: dict[int, list[torch.Tensor]] = {}
        self.covs: dict[int, torch.Tensor] = {}
        self.sums: dict[int, torch.Tensor] = {}
        self.counts: dict[int, int] = {}
        self.context: dict[str, Any] = {}
        self.gpu_cache: dict[tuple[int, str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}
        self.active = (mode == "collect")
        for i, layer in enumerate(_layers(model_kind, model)):
            self.handles.append(layer.register_forward_pre_hook(self._hook(i), with_kwargs=True))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def set_context(self, **kw):
        self.context = kw
        self.active = True

    def deactivate(self):
        self.active = False

    def _visual_mask(self, hidden: torch.Tensor) -> torch.Tensor:
        n = hidden.shape[1]
        if self.model_kind == "qwen":
            mm = self.context.get("mm_token_type_ids")
            if mm is None:
                raise RuntimeError("missing mm_token_type_ids")
            mm = mm[0] if mm.ndim == 2 else mm
            if mm.numel() == n:
                return (mm.to(hidden.device) == 1)
            # During cached decode q_len may be 1; no visual rows in current hidden.
            return torch.zeros(n, dtype=torch.bool, device=hidden.device)
        input_ids = self.context.get("input_ids")
        image_token_id = int(self.context.get("image_token_id"))
        if input_ids is None:
            raise RuntimeError("missing input_ids")
        ids = input_ids[0] if input_ids.ndim == 2 else input_ids
        pos = (ids == image_token_id).nonzero(as_tuple=False).flatten()
        if pos.numel() == 0:
            return torch.zeros(n, dtype=torch.bool, device=hidden.device)
        image_pos = int(pos[0].item())
        n_image_tokens = int(pos.numel())
        n_features = n - (int(ids.numel()) - n_image_tokens)
        if n_features <= 0:
            return torch.zeros(n, dtype=torch.bool, device=hidden.device)
        mask = torch.zeros(n, dtype=torch.bool, device=hidden.device)
        mask[image_pos:image_pos + n_features] = True
        return mask

    def _hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            if not self.active:
                return args, kwargs
            h = kwargs.get("hidden_states", args[0] if args else None)
            if h is None or h.ndim != 3:
                return args, kwargs
            mask = self._visual_mask(h)
            if not bool(mask.any().item()):
                return args, kwargs
            v = h[:, mask, :]
            if self.mode == "collect_cov":
                flat = v[0].detach().float()
                d = int(flat.shape[-1])
                if layer_idx not in self.covs:
                    self.covs[layer_idx] = torch.zeros((d, d), device=flat.device, dtype=torch.float32)
                    self.sums[layer_idx] = torch.zeros((d,), device=flat.device, dtype=torch.float32)
                self.covs[layer_idx].addmm_(flat.T, flat)
                self.sums[layer_idx].add_(flat.sum(dim=0))
                self.counts[layer_idx] = self.counts.get(layer_idx, 0) + int(flat.shape[0])
                return args, kwargs
            if self.mode == "collect":
                remaining = self.max_rows - sum(x.shape[0] for x in self.rows.get(layer_idx, []))
                if remaining > 0:
                    flat = v[0].detach()
                    take = min(remaining, flat.shape[0])
                    if take < flat.shape[0]:
                        idx = torch.randperm(flat.shape[0], device=flat.device)[:take]
                        flat = flat.index_select(0, idx)
                    self.rows.setdefault(layer_idx, []).append(flat.float().cpu())
                self.counts[layer_idx] = self.counts.get(layer_idx, 0) + int(v.shape[1])
                return args, kwargs
            if self.mode == "project" and self.rank is not None and self.rank > 0:
                cache_key = (layer_idx, str(h.device), self.rank)
                cached = self.gpu_cache.get(cache_key)
                if cached is None:
                    entry = self.basis[str(layer_idx)] if str(layer_idx) in self.basis else self.basis[layer_idx]
                    mu_cached = entry["mean"].to(device=h.device, dtype=torch.float32).view(1, 1, -1).contiguous()
                    B_cached = entry["basis"][:, : self.rank].to(device=h.device, dtype=torch.float32).contiguous()
                    cached = (mu_cached, B_cached)
                    self.gpu_cache[cache_key] = cached
                mu, B = cached
                vf = v.float()
                vp = ((vf - mu) @ B) @ B.T + mu
                new_h = h.clone()
                new_h[:, mask, :] = vp.to(dtype=h.dtype)
                if "hidden_states" in kwargs:
                    kwargs = dict(kwargs)
                    kwargs["hidden_states"] = new_h
                    return args, kwargs
                args = (new_h,) + tuple(args[1:])
                return args, kwargs
            return args, kwargs
        return hook


def load_model(model_kind: str, model_path: str, dtype: str, attn: str, device: torch.device):
    dt = getattr(torch, dtype)
    if model_kind == "qwen":
        return load_frozen_qwen3vl(model_path, dt, device, attn)
    return load_frozen_llava(model_path, dt, str(device), attn)


@torch.inference_mode()
def collect_basis(args):
    device = torch.device("cuda:0")
    processor, model = load_model(args.model_kind, args.model_path, args.dtype, args.attn_implementation, device)
    ds = _dataset(args.model_kind, processor, args.benchmark, args.samples, args.data)
    hook = VisualChannelHook(args.model_kind, model, mode="collect", max_rows=args.max_rows)
    image_token_id = int(getattr(model.config, "image_token_index", getattr(processor, "image_token_id", 32000))) if args.model_kind == "llava" else None
    for i in range(len(ds)):
        item = ds[i]
        inp = _to_device_item(item, device)
        if args.model_kind == "qwen":
            hook.set_context(mm_token_type_ids=inp["mm_token_type_ids"])
            if hasattr(model.model, "rope_deltas"):
                model.model.rope_deltas = None
            model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], pixel_values=inp["pixel_values"],
                  image_grid_thw=inp["image_grid_thw"], mm_token_type_ids=inp["mm_token_type_ids"], use_cache=False,
                  return_dict=True, logits_to_keep=1)
        else:
            hook.set_context(input_ids=inp["input_ids"], image_token_id=image_token_id)
            kw = dict(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], pixel_values=inp["pixel_values"],
                      use_cache=False, return_dict=True)
            if "image_sizes" in inp:
                kw["image_sizes"] = inp["image_sizes"]
            model(**kw)
        if (i + 1) % args.log_every == 0:
            print(f"collect {args.model_kind}/{args.benchmark} {i+1}/{len(ds)}", flush=True)
    hook.close()
    out = {"model_kind": args.model_kind, "benchmark": args.benchmark, "samples": len(ds), "max_rows": args.max_rows,
           "max_rank": args.max_rank, "layers": {}, "created": time.time()}
    for l in sorted(hook.rows):
        X = torch.cat(hook.rows[l], dim=0).float()
        mean = X.mean(dim=0)
        Xc = X - mean
        q = min(args.max_rank, Xc.shape[0], Xc.shape[1])
        # GPU SVD is much faster for these moderate sketches.
        U, S, Vh = torch.linalg.svd(Xc.to(device), full_matrices=False)
        basis = Vh[:q].T.contiguous().cpu()
        energy = (S[:q].float().cpu() ** 2)
        out["layers"][str(l)] = {"mean": mean.cpu(), "basis": basis, "energy": energy, "rows": int(X.shape[0]), "seen_rows": hook.counts.get(l, 0)}
        print(f"basis layer {l}: rows={X.shape[0]} q={q}", flush=True)
        del X, Xc, U, S, Vh, basis
        torch.cuda.empty_cache()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    print(f"saved basis {path}", flush=True)


@torch.inference_mode()
def generate_fresh(model_kind: str, model, processor, item: dict, inp: dict, hook: VisualChannelHook | None,
                   max_new_tokens: int, device: torch.device, image_token_id: int | None = None):
    full_ids = inp["input_ids"].clone()
    full_mask = inp["attention_mask"].clone()
    generated=[]
    eos = set()
    tok = processor.tokenizer
    if getattr(tok, "eos_token_id", None) is not None:
        eos.add(int(tok.eos_token_id))
    extra = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(extra, list): eos.update(int(x) for x in extra)
    elif extra is not None: eos.add(int(extra))
    text = ""
    for _ in range(max_new_tokens):
        if model_kind == "qwen":
            # mm ids grow with generated text zeros
            mm = inp["mm_token_type_ids"]
            if full_ids.shape[1] != mm.shape[1]:
                add = full_ids.shape[1] - mm.shape[1]
                mm = torch.cat([mm, torch.zeros((1, add), dtype=mm.dtype, device=mm.device)], dim=1)
            if hook is not None:
                hook.set_context(mm_token_type_ids=mm)
            if hasattr(model.model, "rope_deltas"):
                model.model.rope_deltas = None
            out = model(input_ids=full_ids, attention_mask=full_mask, pixel_values=inp["pixel_values"],
                        image_grid_thw=inp["image_grid_thw"], mm_token_type_ids=mm, use_cache=False,
                        return_dict=True, logits_to_keep=1)
        else:
            if hook is not None:
                hook.set_context(input_ids=full_ids, image_token_id=image_token_id)
            kw = dict(input_ids=full_ids, attention_mask=full_mask, pixel_values=inp["pixel_values"], use_cache=False, return_dict=True)
            if "image_sizes" in inp:
                kw["image_sizes"] = inp["image_sizes"]
            out = model(**kw)
        next_token = int(out.logits[0, -1].float().argmax().item())
        generated.append(next_token)
        text = tok.decode(generated, skip_special_tokens=True).strip()
        if next_token in eos or extract_option_from_text(text) in ["A", "B", "C", "D"]:
            break
        t = torch.tensor([[next_token]], dtype=full_ids.dtype, device=device)
        full_ids = torch.cat([full_ids, t], dim=1)
        full_mask = torch.cat([full_mask, torch.ones_like(t)], dim=1)
    return text


def _load_basis(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["layers"]


@torch.inference_mode()
def eval_worker(args):
    local = 0
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    processor, model = load_model(args.model_kind, args.model_path, args.dtype, args.attn_implementation, device)
    ds = _dataset(args.model_kind, processor, args.benchmark, args.samples, args.data)
    total = len(ds)
    indices = list(range(int(args.shard), total, int(args.world)))
    spec = get_benchmark_spec(args.benchmark)
    ranks = [None if r == "native" else int(r) for r in args.ranks]
    basis = _load_basis(args.basis) if any(r is not None for r in ranks) else None
    hooks = {r: (None if r is None else VisualChannelHook(args.model_kind, model, mode="project", rank=r, basis=basis)) for r in ranks}
    image_token_id = int(getattr(model.config, "image_token_index", getattr(processor, "image_token_id", 32000))) if args.model_kind == "llava" else None
    rows=[]
    correct={"native" if r is None else f"r{r}":0 for r in ranks}
    for c, idx in enumerate(indices):
        item = ds[idx]
        inp = _to_device_item(item, device)
        rec={"index": item.get("index", idx), "row": idx, "outputs": {}}
        for r in ranks:
            for _h in hooks.values():
                if _h is not None:
                    _h.deactivate()
            name = "native" if r is None else f"r{r}"
            text = generate_fresh(args.model_kind, model, processor, item, inp, hooks[r], spec.max_new_tokens, device, image_token_id)
            scored = score_prediction(metric=spec.metric, prediction_text=text, answer=item.get("answer"), answers=item.get("answers"), choices=item.get("choices"))
            rec["outputs"][name] = {"text": text, "score": scored}
            correct[name] += int(float(scored.get("score", 0.0)) > 0.0)
        rows.append(rec)
        if (c+1) % args.log_every == 0:
            print(f"eval {args.model_kind}/{args.benchmark} shard {args.shard} {c+1}/{len(indices)}", flush=True)
    for h in hooks.values():
        if h is not None: h.close()
    out={"model_kind":args.model_kind,"benchmark":args.benchmark,"samples":len(indices),"shard":args.shard,"world":args.world,
         "correct":correct,"rows":rows}
    path=Path(args.output)/f"{args.model_kind}_{args.benchmark}_shard{args.shard}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {path}", flush=True)


def aggregate(root: Path, model_kind: str, benchmark: str, world: int):
    files=[root/f"{model_kind}_{benchmark}_shard{s}.json" for s in range(world)]
    data=[json.loads(p.read_text()) for p in files]
    total=sum(d["samples"] for d in data)
    names=list(data[0]["correct"].keys())
    summary={n:{"correct":sum(d["correct"].get(n,0) for d in data),"samples":total} for n in names}
    for n in names:
        summary[n]["accuracy_pct"]=100.0*summary[n]["correct"]/max(1,total)
    return {"model_kind":model_kind,"benchmark":benchmark,"samples":total,"conditions":summary}



@torch.inference_mode()
def collect_cov_worker(args):
    local = 0
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    processor, model = load_model(args.model_kind, args.model_path, args.dtype, args.attn_implementation, device)
    ds = _dataset(args.model_kind, processor, args.benchmark, args.samples, args.data)
    hook = VisualChannelHook(args.model_kind, model, mode="collect_cov")
    image_token_id = int(getattr(model.config, "image_token_index", getattr(processor, "image_token_id", 32000))) if args.model_kind == "llava" else None
    total = len(ds)
    indices = list(range(int(args.shard), total, int(args.world)))
    for c, idx in enumerate(indices):
        item = ds[idx]
        inp = _to_device_item(item, device)
        if args.model_kind == "qwen":
            hook.set_context(mm_token_type_ids=inp["mm_token_type_ids"])
            if hasattr(model.model, "rope_deltas"):
                model.model.rope_deltas = None
            model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], pixel_values=inp["pixel_values"],
                  image_grid_thw=inp["image_grid_thw"], mm_token_type_ids=inp["mm_token_type_ids"], use_cache=False,
                  return_dict=True, logits_to_keep=1)
        else:
            hook.set_context(input_ids=inp["input_ids"], image_token_id=image_token_id)
            kw = dict(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], pixel_values=inp["pixel_values"],
                      use_cache=False, return_dict=True)
            if "image_sizes" in inp:
                kw["image_sizes"] = inp["image_sizes"]
            model(**kw)
        if (c + 1) % args.log_every == 0:
            print(f"collect-cov {args.model_kind}/{args.benchmark} shard {args.shard} {c+1}/{len(indices)}", flush=True)
    out = {"model_kind": args.model_kind, "benchmark": args.benchmark, "samples": len(indices), "shard": args.shard,
           "world": args.world, "layers": {}, "created": time.time()}
    for l in sorted(hook.covs):
        out["layers"][str(l)] = {"cov": hook.covs[l].cpu(), "sum": hook.sums[l].cpu(), "count": hook.counts[l]}
    path = Path(args.output) / f"{args.model_kind}_{args.benchmark}_cov_shard{args.shard}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    print(f"saved {path}", flush=True)


def merge_cov_basis(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = Path(args.input)
    shards = [torch.load(root / f"{args.model_kind}_{args.benchmark}_cov_shard{s}.pt", map_location="cpu", weights_only=False) for s in range(args.world)]
    layer_keys = sorted(shards[0]["layers"], key=lambda x: int(x))
    out = {"model_kind": args.model_kind, "benchmark": args.benchmark, "samples": sum(s["samples"] for s in shards),
           "max_rank": args.max_rank, "basis_source": "exact_streaming_covariance_all_visual_rows", "layers": {}, "created": time.time()}
    for lk in layer_keys:
        cov = sum(s["layers"][lk]["cov"] for s in shards)
        sm = sum(s["layers"][lk]["sum"] for s in shards)
        cnt = sum(int(s["layers"][lk]["count"]) for s in shards)
        mean = sm / max(1, cnt)
        centered = cov - torch.outer(sm, sm) / max(1, cnt)
        centered = (centered + centered.T).mul_(0.5)
        evals, evecs = torch.linalg.eigh(centered.to(device))
        q = min(args.max_rank, evecs.shape[1])
        basis = evecs[:, -q:].flip(1).contiguous().cpu()
        energy = evals[-q:].flip(0).float().cpu().clamp_min_(0)
        out["layers"][lk] = {"mean": mean.float().cpu(), "basis": basis, "energy": energy, "rows": cnt, "seen_rows": cnt}
        print(f"basis {args.model_kind}/{args.benchmark} layer {lk}: rows={cnt} q={q}", flush=True)
        del cov, sm, centered, evals, evecs
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    print(f"saved basis {path}", flush=True)


def launch_collect_cov(args):
    root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
    procs=[]; gpus=[int(x) for x in args.gpus]
    for shard,gpu in enumerate(gpus):
        cmd=[sys.executable,"-u","-m","src.visual_channel_rank_grid","collect-cov-worker","--model-kind",args.model_kind,"--model-path",args.model_path,
             "--benchmark",args.benchmark,"--samples",str(args.samples),"--output",str(root),"--world",str(len(gpus)),"--shard",str(shard),"--gpu",str(gpu),
             "--dtype",args.dtype,"--attn-implementation",args.attn_implementation,"--log-every",str(args.log_every)]
        if args.data: cmd += ["--data", args.data]
        env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
        log = root / f"{args.model_kind}_{args.benchmark}_cov_shard{shard}.log"
        f=open(log,"w")
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        procs.append((shard,p,log,time.time()))
        print("START", shard, gpu, p.pid, log, flush=True)
    bad=[]
    for shard,p,log,t0 in procs:
        rc=p.wait(); print("DONE", shard, rc, "elapsed", round(time.time()-t0,1), log, flush=True)
        if rc: bad.append((shard,rc,log))
    if bad: raise SystemExit(f"failed collect shards {bad}")
    merge_args=argparse.Namespace(input=str(root), output=args.basis_out, model_kind=args.model_kind, benchmark=args.benchmark, world=len(gpus), max_rank=args.max_rank)
    merge_cov_basis(merge_args)

def launch_eval(args):
    root=Path(args.output); root.mkdir(parents=True, exist_ok=True)
    procs=[]
    gpus=[int(x) for x in args.gpus]
    for shard,gpu in enumerate(gpus):
        cmd=[sys.executable,"-u","-m","src.visual_channel_rank_grid","eval-worker","--model-kind",args.model_kind,"--model-path",args.model_path,
             "--benchmark",args.benchmark,"--samples",str(args.samples),"--basis",args.basis,"--output",str(root),"--world",str(len(gpus)),"--shard",str(shard),"--gpu",str(gpu),
             "--dtype",args.dtype,"--attn-implementation",args.attn_implementation,"--ranks",*args.ranks]
        if args.data: cmd += ["--data", args.data]
        env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
        p=subprocess.Popen(cmd,cwd=ROOT,env=env)
        procs.append((shard,p))
        print("START", shard, gpu, p.pid, flush=True)
    bad=[]
    for shard,p in procs:
        rc=p.wait(); print("DONE", shard, rc, flush=True)
        if rc: bad.append((shard,rc))
    if bad: raise SystemExit(f"failed shards {bad}")
    summary=aggregate(root,args.model_kind,args.benchmark,len(gpus))
    (root/f"{args.model_kind}_{args.benchmark}_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2), flush=True)


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="cmd", required=True)
    c=sub.add_parser("collect-basis")
    c.add_argument("--model-kind", choices=["qwen","llava"], required=True)
    c.add_argument("--model-path", required=True)
    c.add_argument("--benchmark", choices=["sqa","realworldqa","mmstar"], required=True)
    c.add_argument("--data")
    c.add_argument("--samples", type=int, default=128)
    c.add_argument("--max-rows", type=int, default=2048)
    c.add_argument("--max-rank", type=int, default=1024)
    c.add_argument("--output", required=True)
    c.add_argument("--dtype", default="bfloat16")
    c.add_argument("--attn-implementation", default="flash_attention_2")
    c.add_argument("--log-every", type=int, default=20)
    e=sub.add_parser("eval-worker")
    e.add_argument("--model-kind", choices=["qwen","llava"], required=True)
    e.add_argument("--model-path", required=True)
    e.add_argument("--benchmark", choices=["sqa","realworldqa","mmstar"], required=True)
    e.add_argument("--data")
    e.add_argument("--samples", type=int, default=1000)
    e.add_argument("--basis", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--world", type=int, default=1); e.add_argument("--shard", type=int, default=0); e.add_argument("--gpu", type=int, default=0)
    e.add_argument("--ranks", nargs="+", default=["native","32","64","128","256","512","1024"])
    e.add_argument("--dtype", default="bfloat16"); e.add_argument("--attn-implementation", default="flash_attention_2")
    e.add_argument("--log-every", type=int, default=20)
    cw=sub.add_parser("collect-cov-worker")
    cw.add_argument("--model-kind", choices=["qwen","llava"], required=True)
    cw.add_argument("--model-path", required=True)
    cw.add_argument("--benchmark", choices=["sqa","realworldqa","mmstar"], required=True)
    cw.add_argument("--data")
    cw.add_argument("--samples", type=int, default=1000)
    cw.add_argument("--output", required=True)
    cw.add_argument("--world", type=int, default=1); cw.add_argument("--shard", type=int, default=0); cw.add_argument("--gpu", type=int, default=0)
    cw.add_argument("--dtype", default="bfloat16"); cw.add_argument("--attn-implementation", default="flash_attention_2")
    cw.add_argument("--log-every", type=int, default=20)
    lc=sub.add_parser("launch-collect-cov")
    lc.add_argument("--model-kind", choices=["qwen","llava"], required=True)
    lc.add_argument("--model-path", required=True)
    lc.add_argument("--benchmark", choices=["sqa","realworldqa","mmstar"], required=True)
    lc.add_argument("--data")
    lc.add_argument("--samples", type=int, default=1000)
    lc.add_argument("--output-dir", required=True)
    lc.add_argument("--basis-out", required=True)
    lc.add_argument("--gpus", nargs="+", default=["0"])
    lc.add_argument("--max-rank", type=int, default=1024)
    lc.add_argument("--dtype", default="bfloat16"); lc.add_argument("--attn-implementation", default="flash_attention_2")
    lc.add_argument("--log-every", type=int, default=20)
    mb=sub.add_parser("merge-cov-basis")
    mb.add_argument("--input", required=True); mb.add_argument("--output", required=True)
    mb.add_argument("--model-kind", choices=["qwen","llava"], required=True)
    mb.add_argument("--benchmark", choices=["sqa","realworldqa","mmstar"], required=True)
    mb.add_argument("--world", type=int, required=True); mb.add_argument("--max-rank", type=int, default=1024)
    l=sub.add_parser("launch-eval")
    l.add_argument("--model-kind", choices=["qwen","llava"], required=True)
    l.add_argument("--model-path", required=True)
    l.add_argument("--benchmark", choices=["sqa","realworldqa","mmstar"], required=True)
    l.add_argument("--data")
    l.add_argument("--samples", type=int, default=1000)
    l.add_argument("--basis", required=True); l.add_argument("--output", required=True)
    l.add_argument("--gpus", nargs="+", default=["0"])
    l.add_argument("--ranks", nargs="+", default=["native","32","64","128","256","512","1024"])
    l.add_argument("--dtype", default="bfloat16"); l.add_argument("--attn-implementation", default="flash_attention_2")
    a=p.parse_args()
    if a.cmd=="collect-basis": collect_basis(a)
    elif a.cmd=="collect-cov-worker": collect_cov_worker(a)
    elif a.cmd=="launch-collect-cov": launch_collect_cov(a)
    elif a.cmd=="merge-cov-basis": merge_cov_basis(a)
    elif a.cmd=="eval-worker": eval_worker(a)
    elif a.cmd=="launch-eval": launch_eval(a)

if __name__ == "__main__": main()
