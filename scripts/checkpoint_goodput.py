#!/usr/bin/env python3
"""Estimate checkpoint-stall-adjusted training goodput."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def get_list(data: dict[str, Any], *keys: str) -> list[float]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return [float(x) for x in value]
    return []


def compute(data: dict[str, Any]) -> dict[str, Any]:
    step_times = get_list(data, "step_times", "step_times_sec")
    checkpoint_times = get_list(data, "checkpoint_times", "checkpoint_times_sec")
    units_per_step = float(data.get("units_per_step", data.get("tokens_per_step", data.get("samples_per_step", 1.0))))
    total_units = units_per_step * len(step_times)
    training_time = sum(step_times)
    checkpoint_time = sum(checkpoint_times)
    wall_time = training_time + checkpoint_time
    return {
        "steps": len(step_times),
        "units_per_step": units_per_step,
        "total_units": total_units,
        "training_time_sec": training_time,
        "checkpoint_blocking_time_sec": checkpoint_time,
        "wall_time_with_checkpoints_sec": wall_time,
        "raw_throughput_units_per_sec": total_units / training_time if training_time else None,
        "checkpoint_adjusted_goodput_units_per_sec": total_units / wall_time if wall_time else None,
        "checkpoint_overhead_fraction": checkpoint_time / wall_time if wall_time else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_json", type=Path, help="JSON with step_times and optional checkpoint_times.")
    args = parser.parse_args()
    data = json.loads(args.timing_json.read_text(encoding="utf-8"))
    print(json.dumps(compute(data), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
