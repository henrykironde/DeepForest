from __future__ import annotations

import csv
import os
import time
from collections.abc import Iterable
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

    def _read_csv_rows(self) -> tuple[list[str], list[dict[str, str]]]:
        if not os.path.exists(self.log_file):
            return [], []

        with open(self.log_file, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return [], []
            rows = [dict(row) for row in reader]
            return list(reader.fieldnames), rows

    def _write_csv_rows(
        self, fieldnames: list[str], rows: Iterable[dict[str, object]]
    ) -> None:
        os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
        with open(self.log_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    def log(self, payload: dict) -> None:
        payload = dict(payload)
        payload["timestamp"] = datetime.now(UTC).isoformat()

        existing_fieldnames, existing_rows = self._read_csv_rows()
        new_keys = list(payload.keys())

        if not existing_fieldnames:
            # Stable ordering: timestamp first, then stage/batch_idx if present, then the rest.
            preferred = ["timestamp", "stage", "batch_idx"]
            fieldnames = []
            for k in preferred:
                if k in payload and k not in fieldnames:
                    fieldnames.append(k)
            for k in sorted(new_keys):
                if k not in fieldnames:
                    fieldnames.append(k)
            self._write_csv_rows(fieldnames, [payload])
            return

        # If new keys appear later (e.g., different GPU visibility), expand header by rewriting file.
        missing_in_header = [k for k in new_keys if k not in existing_fieldnames]
        if missing_in_header:
            fieldnames = list(existing_fieldnames) + missing_in_header
            self._write_csv_rows(fieldnames, [*existing_rows, payload])
            return

        # Fast append path: header already contains all keys
        with open(self.log_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=existing_fieldnames, extrasaction="ignore"
            )
            writer.writerow({k: payload.get(k, "") for k in existing_fieldnames})


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
