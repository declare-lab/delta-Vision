"""Summarize measured per-request CUDA memory in MiB (never sum peaks)."""
import statistics


def summarize_peak_memory(predictions, *, prefix=''):
    def values(key):
        return [p[prefix + key] for p in predictions if p.get(prefix + key) is not None]

    peaks, reserved = [], []
    for p in predictions:
        for target, direct, fields in [
            (peaks, 'peak_memory_mb', ('prefill_peak_allocated_mb', 'decode_peak_allocated_mb')),
            (reserved, 'peak_reserved_mb', ('prefill_peak_reserved_mb', 'decode_peak_reserved_mb')),
        ]:
            measured = [p[prefix + k] for k in fields if p.get(prefix + k) is not None]
            if p.get(prefix + direct) is not None:
                target.append(p[prefix + direct])
            elif measured:
                target.append(max(measured))
    prefill = values('prefill_peak_allocated_mb')
    decode = [p[prefix + 'decode_peak_allocated_mb'] for p in predictions
              if p.get(prefix + 'decode_peak_allocated_mb') is not None
              and p.get(prefix + 'decode_steps', 1) != 0]
    return dict(peak_memory_mb=max(peaks) if peaks else None,
        peak_memory_mb_mean=statistics.mean(peaks) if peaks else None,
        peak_reserved_mb=max(reserved) if reserved else None,
        prefill_peak_memory_mb=max(prefill) if prefill else None,
        decode_peak_memory_mb=max(decode) if decode else None,
        peak_memory_samples=len(peaks))
