"""Observe actual prefill and cached decode forwards inside native generate."""
import time

import torch


class GenerationStageTimer:
    def __init__(self, model, *, measure_memory=False):
        self.device = next(model.parameters()).device
        self.measure_memory = measure_memory and self.device.type == 'cuda'
        self.active = False
        self.stages = []
        self.handles = [model.register_forward_pre_hook(self._before, with_kwargs=True),
                        model.register_forward_hook(self._after, with_kwargs=True)]

    def begin(self):
        self.stages = []
        self.active = True
        self.memory = {}
        self.request_start_s = None
        self.request_prefill_s = None
        if self.measure_memory:
            torch.cuda.reset_peak_memory_stats(self.device)
            self.memory_start = torch.cuda.memory_allocated(self.device) / 1024**2

    def mark_request_start(self, timestamp):
        """Use the outer request clock, including generation/position setup."""
        self.request_start_s = timestamp

    def _before(self, module, args, kwargs):
        if not self.active:
            return
        ids = kwargs.get("input_ids")
        self.current = {"kind": "prefill" if not self.stages else "decode",
                        "input_tokens": int(ids.shape[-1]) if ids is not None else int(kwargs["inputs_embeds"].shape[-2]),
                        "has_pixels": kwargs.get("pixel_values") is not None or kwargs.get("pixel_values_videos") is not None}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.start = time.perf_counter()

    def _after(self, module, args, kwargs, output):
        if not self.active:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        end = time.perf_counter()
        self.current["seconds"] = end - self.start
        if self.current["kind"] == "prefill":
            if self.request_start_s is not None:
                self.request_prefill_s = end - self.request_start_s
            if self.measure_memory:
                self.memory['prefill_peak_allocated_mb'] = torch.cuda.max_memory_allocated(self.device) / 1024**2
                self.memory['prefill_peak_reserved_mb'] = torch.cuda.max_memory_reserved(self.device) / 1024**2
                torch.cuda.reset_peak_memory_stats(self.device)
            cache = output.past_key_values
            self.current["cache_lengths"] = [int(layer.get_seq_length()) for layer in cache.layers]
            seen = set()
            nbytes = 0
            for layer in cache.layers:
                for tensor in (layer.keys, layer.values):
                    storage = tensor.untyped_storage()
                    key = (str(tensor.device), storage.data_ptr())
                    if key not in seen:
                        nbytes += storage.nbytes()
                        seen.add(key)
            self.current["kv_cache_mib"] = nbytes / 1024**2
        self.stages.append(self.current)

    def finish(self, total_s, generated_tokens):
        self.active = False
        prefill = sum(s["seconds"] for s in self.stages if s["kind"] == "prefill")
        decode = sum(s["seconds"] for s in self.stages if s["kind"] == "decode")
        steps = sum(s["kind"] == "decode" for s in self.stages)
        assert steps == generated_tokens - 1, (steps, generated_tokens)
        assert all(s["input_tokens"] == 1 and not s["has_pixels"] for s in self.stages[1:])
        if self.measure_memory:
            continuation = torch.cuda.max_memory_allocated(self.device) / 1024**2
            reserved = torch.cuda.max_memory_reserved(self.device) / 1024**2
            self.memory.update(decode_peak_allocated_mb=continuation if steps else None,
                decode_peak_reserved_mb=reserved if steps else None,
                peak_memory_mb=max(self.memory['prefill_peak_allocated_mb'], continuation),
                peak_reserved_mb=max(self.memory['prefill_peak_reserved_mb'], reserved),
                request_start_allocated_mb=self.memory_start)
            self.memory['peak_memory_delta_mb'] = max(0., self.memory['peak_memory_mb'] - self.memory_start)
        request_prefill = ({"request_prefill_time_s": self.request_prefill_s}
                           if self.request_prefill_s is not None else {})
        return {"generation_prefill_time_s": prefill, "decode_time_s": decode,
                "generation_overhead_s": total_s - prefill - decode,
                "decode_steps": steps, "generated_tokens": generated_tokens,
                "actual_prefill_kv_cache_mb": self.stages[0]["kv_cache_mib"],
                "generation_stages": self.stages, **self.memory, **request_prefill}

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
