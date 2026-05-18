#!/usr/bin/env python3
"""Sweep PyTorch DataLoader worker/prefetch settings.

The factory should be a Python callable referenced as module:function. It must
return either a Dataset or (dataset, collate_fn).

Example:
  python scripts/dataloader_sweep.py --factory mypkg.data:make_dataset --batch-size 32
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_factory(spec: str):
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def build_dataset(factory_spec: str):
    value = load_factory(factory_spec)()
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("Factory tuple must be (dataset, collate_fn)")
        return value
    return value, None


def run_once(dataset: Any, collate_fn: Any, batch_size: int, num_workers: int, prefetch_factor: int, pin_memory: bool, persistent_workers: bool, batches: int) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(**kwargs)

    it = iter(loader)
    times = []
    for _ in range(batches):
        start = time.perf_counter()
        try:
            next(it)
        except StopIteration:
            break
        times.append(time.perf_counter() - start)
    return {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        "batches": len(times),
        "mean_batch_fetch_time": sum(times) / len(times) if times else None,
        "min_batch_fetch_time": min(times) if times else None,
        "max_batch_fetch_time": max(times) if times else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True, help="module:function returning dataset or (dataset, collate_fn).")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", default="0,2,4,8")
    parser.add_argument("--prefetch-factors", default="2,4")
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset, collate_fn = build_dataset(args.factory)
    results = []
    for num_workers in parse_csv_ints(args.num_workers):
        prefetch_values = parse_csv_ints(args.prefetch_factors) if num_workers > 0 else [2]
        for prefetch_factor in prefetch_values:
            for pin_memory in (False, True):
                for persistent_workers in (False, True):
                    if num_workers == 0 and persistent_workers:
                        continue
                    results.append(run_once(dataset, collate_fn, args.batch_size, num_workers, prefetch_factor, pin_memory, persistent_workers, args.batches))
    text = json.dumps({"results": results}, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
