"""
gpu_stats.py
============
Best-effort GPU utilisation / memory metrics for the response payload.

Order of preference:
  1. pynvml (NVIDIA Management Library) — gives both util% and mem%
  2. torch.cuda  — gives mem% only (no util)
  3. None — returns zeros, never crashes the runtime
"""

from __future__ import annotations

_pynvml_handle = None
_pynvml_failed = False


def _try_pynvml() -> tuple[float, float] | None:
    global _pynvml_handle, _pynvml_failed
    if _pynvml_failed:
        return None
    try:
        import pynvml  # type: ignore
    except ImportError:
        _pynvml_failed = True
        return None

    try:
        if _pynvml_handle is None:
            pynvml.nvmlInit()
            _pynvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(_pynvml_handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(_pynvml_handle)
        mem_pct = (mem.used / mem.total) * 100.0 if mem.total > 0 else 0.0
        return float(mem_pct), float(util.gpu)
    except Exception:
        _pynvml_failed = True
        return None


def _try_torch() -> tuple[float, float] | None:
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        used = total - free
        mem_pct = (used / total) * 100.0 if total > 0 else 0.0
        return float(mem_pct), 0.0
    except Exception:
        return None


def gpu_stats() -> tuple[float, float]:
    """Return ``(mem_percent, util_percent)``. Either may be 0.0 if unavailable."""
    res = _try_pynvml()
    if res is not None:
        return res
    res = _try_torch()
    if res is not None:
        return res
    return 0.0, 0.0
