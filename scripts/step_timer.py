#!/usr/bin/env python3
"""Summarize steady-state PyTorch training step times.

This file is both a tiny library and a CLI. Import StepTimer into a training
loop, or pass a JSON file containing a list of step durations.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable


class StepTimer:
    """Context manager for timing CUDA or CPU training steps.

    Example:
        timer = StepTimer(device="cuda", warmup_steps=10)
        for step, batch in enumerate(loader):
            with timer.time_step():
                loss = train_step(batch)
            if timer.measured_steps >= 50:
                break
        print(timer.summary())
    """

    def __init__(self, device: str = "cuda", warmup_steps: int = 10) -> None:
        self.device = device
        self.warmup_steps = warmup_steps
        self.total_steps = 0
        self.times: list[float] = []

    @property
    def measured_steps(self) -> int:
        return len(self.times)

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass

    class _Ctx:
        def __init__(self, parent: "StepTimer") -> None:
            self.parent = parent
            self.start = 0.0

        def __enter__(self) -> None:
            if self.parent.total_steps >= self.parent.warmup_steps:
                self.parent._sync()
                self.start = time.perf_counter()
            return None

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            if self.parent.total_steps >= self.parent.warmup_steps:
                self.parent._sync()
                self.parent.times.append(time.perf_counter() - self.start)
            self.parent.total_steps += 1

    def time_step(self) -> "StepTimer._Ctx":
        return StepTimer._Ctx(self)

    def summary(self) -> dict[str, float | int | None]:
        return summarize(self.times)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[int(pos)]
    return sorted_values[lower] * (upper - pos) + sorted_values[upper] * (pos - lower)


def summarize(times: Iterable[float]) -> dict[str, float | int | None]:
    vals = [float(x) for x in times]
    if not vals:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "std": None, "min": None, "max": None}
    return {
        "count": len(vals),
        "mean": statistics.fmean(vals),
        "p50": percentile(vals, 0.50),
        "p95": percentile(vals, 0.95),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def load_times(path: Path) -> list[float]:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [float(x) for x in data]
    if isinstance(data, dict):
        for key in ("step_times", "step_times_sec", "times", "durations"):
            if key in data:
                return [float(x) for x in data[key]]
    raise ValueError(f"Could not find step times in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--times-file", type=Path, help="JSON list or object containing step_times.")
    parser.add_argument("--throughput-units", type=float, default=None, help="Samples/tokens/images per step.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    if not args.times_file:
        parser.print_help()
        return 2

    times = load_times(args.times_file)
    summary = summarize(times)
    if args.throughput_units and summary["mean"]:
        summary["throughput_per_sec"] = args.throughput_units / float(summary["mean"])
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
