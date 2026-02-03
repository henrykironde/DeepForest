from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import psutil
import torch


@dataclass(frozen=True)
class ResourceSnapshot:
    rss_mb: float
    gpu_allocated_mb: dict[int, float]
    gpu_reserved_mb: dict[int, float]
    gpu_max_allocated_mb: dict[int, float]


class ResourceTracker:
    """Lightweight per-step resource logger (CPU + GPU).

    Enable by setting environment variable:
      DEEPFOREST_BENCHMARK_LOG=/path/to/logfile.log

    In multi-process (DDP) settings, each rank should write to a separate file.
    """

    def __init__(self, log_file: str):
        self.log_file = log_file
        self._process = psutil.Process(os.getpid())

    def cpu_rss_mb(self) -> float:
        return self._process.memory_info().rss / (1024**2)

    def snapshot(self) -> ResourceSnapshot:
        rss_mb = self.cpu_rss_mb()
        gpu_allocated_mb: dict[int, float] = {}
        gpu_reserved_mb: dict[int, float] = {}
        gpu_max_allocated_mb: dict[int, float] = {}

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_allocated_mb[i] = torch.cuda.memory_allocated(i) / (1024**2)
                gpu_reserved_mb[i] = torch.cuda.memory_reserved(i) / (1024**2)
                gpu_max_allocated_mb[i] = torch.cuda.max_memory_allocated(i) / (1024**2)

        return ResourceSnapshot(
            rss_mb=rss_mb,
            gpu_allocated_mb=gpu_allocated_mb,
            gpu_reserved_mb=gpu_reserved_mb,
            gpu_max_allocated_mb=gpu_max_allocated_mb,
        )

    def reset_gpu_peaks(self) -> None:
        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

    def log(self, payload: dict) -> None:
        payload = dict(payload)
        payload["timestamp"] = datetime.now(UTC).isoformat()

        line = ",".join(f"{k}={v}" for k, v in payload.items())
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def tracker_from_env() -> ResourceTracker | None:
    """Create a tracker if DEEPFOREST_BENCHMARK_LOG is set."""
    log_file = os.environ.get("DEEPFOREST_BENCHMARK_LOG")
    if not log_file:
        return None

    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    if rank is not None:
        root, ext = os.path.splitext(log_file)
        log_file = f"{root}_rank{rank}{ext or '.log'}"

    return ResourceTracker(log_file=log_file)


def timed_call(fn, *args, **kwargs):
    """Return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0
