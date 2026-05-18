#!/usr/bin/env python3
"""Compare before/after PyTorch training run records.

Expected input is JSON. Unknown fields are allowed. Recommended keys:
  run_conditions: {global_batch_size, microbatch_size, sequence_length, precision, ...}
  metrics: {mean_step_time, p50_step_time, throughput, peak_cuda_memory, ...}
  correctness: {loss_proxy, validation_metric, nan_inf_count, ...}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

INVARIANT_KEYS = [
    "global_batch_size",
    "microbatch_size",
    "gradient_accumulation",
    "sequence_length",
    "input_resolution",
    "dataset",
    "dataset_split",
    "seed",
    "optimizer",
    "scheduler",
    "loss_definition",
    "evaluation_protocol",
    "checkpoint_format",
]

METRIC_KEYS = [
    "mean_step_time",
    "p50_step_time",
    "p95_step_time",
    "throughput",
    "throughput_per_gpu",
    "gpu_utilization",
    "peak_cuda_memory",
    "dataloader_wait",
    "communication_time",
    "checkpoint_blocking_time",
    "goodput",
]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(d: dict[str, Any], section: str, key: str) -> Any:
    value = d.get(section, {})
    return value.get(key) if isinstance(value, dict) else None


def pct_change(before: Any, after: Any) -> str:
    try:
        b = float(before)
        a = float(after)
        if b == 0:
            return "n/a"
        return f"{(a - b) / b * 100:+.2f}%"
    except Exception:
        return "n/a"


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    invariants = []
    for key in INVARIANT_KEYS:
        b = nested_get(before, "run_conditions", key)
        a = nested_get(after, "run_conditions", key)
        if b is not None or a is not None:
            invariants.append({"key": key, "before": b, "after": a, "same": b == a})

    metrics = []
    for key in METRIC_KEYS:
        b = nested_get(before, "metrics", key)
        a = nested_get(after, "metrics", key)
        if b is not None or a is not None:
            metrics.append({"key": key, "before": b, "after": a, "change": pct_change(b, a)})

    correctness = []
    b_corr = before.get("correctness", {}) if isinstance(before.get("correctness", {}), dict) else {}
    a_corr = after.get("correctness", {}) if isinstance(after.get("correctness", {}), dict) else {}
    for key in sorted(set(b_corr) | set(a_corr)):
        correctness.append({"key": key, "before": b_corr.get(key), "after": a_corr.get(key), "same": b_corr.get(key) == a_corr.get(key)})

    return {"invariants": invariants, "metrics": metrics, "correctness": correctness}


def to_markdown(result: dict[str, Any]) -> str:
    lines = ["# Before/After Run Comparison", ""]
    lines += ["## Invariant Checks", "", "| Key | Before | After | Same? |", "| --- | --- | --- | --- |"]
    for row in result["invariants"]:
        lines.append(f"| {row['key']} | {row['before']} | {row['after']} | {row['same']} |")
    lines += ["", "## Metrics", "", "| Metric | Before | After | Change |", "| --- | ---: | ---: | ---: |"]
    for row in result["metrics"]:
        lines.append(f"| {row['key']} | {row['before']} | {row['after']} | {row['change']} |")
    lines += ["", "## Correctness", "", "| Key | Before | After | Same? |", "| --- | --- | --- | --- |"]
    for row in result["correctness"]:
        lines.append(f"| {row['key']} | {row['before']} | {row['after']} | {row['same']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    parser.add_argument("--fail-on-invariant-change", action="store_true")
    args = parser.parse_args()

    result = compare(load(args.before), load(args.after))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(to_markdown(result), end="")

    if args.fail_on_invariant_change and any(not row["same"] for row in result["invariants"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
